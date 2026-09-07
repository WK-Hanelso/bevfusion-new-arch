"""DepthFusion-inspired dense global fusion for canonical BEV features."""

from typing import Sequence

import torch
from torch import nn

from mmdet3d.models.builder import FUSERS
from mmdet3d.models.utils.bev_grid import BEVGridSpec

from .depth_guided_attention import DepthGuidedDeformableAttention2D


def _patch_encoder(in_channels, hidden_channels, out_channels, patch_size):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            hidden_channels,
            kernel_size=patch_size,
            stride=patch_size,
        ),
        nn.BatchNorm2d(hidden_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1),
        nn.ReLU(inplace=True),
    )


@FUSERS.register_module()
class DepthGFusion(nn.Module):
    """Depth-aware global fusion adapted from official DepthFusion DGF.

    The official operator first converts 180x180 BEV features to 36x36 learned
    patches, multiplies LiDAR queries by a radial depth encoding, performs
    deformable cross-attention over camera BEV, then restores the dense grid
    with camera/LiDAR residuals and a convolutional FFN. This implementation
    retains that flow while accepting the existing BEVFusion branch channels.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int = 256,
        embed_channels: int = 256,
        patch_hidden_channels: int = 640,
        num_heads: int = 8,
        patch_size: int = 5,
        attention_downsample: int = 4,
        xbound=(-54.0, 54.0, 0.6),
        ybound=(-54.0, 54.0, 0.6),
        camera_layout: str = "yx",
        lidar_layout: str = "yx",
        dropout: float = 0.0,
    ):
        super().__init__()
        if len(in_channels) != 2:
            raise ValueError("DepthGFusion expects [camera_channels, lidar_channels]")
        if embed_channels % num_heads:
            raise ValueError("embed_channels must be divisible by num_heads")
        self.grid = BEVGridSpec(tuple(xbound), tuple(ybound))
        if self.grid.height % patch_size or self.grid.width % patch_size:
            raise ValueError("BEV dimensions must be divisible by patch_size")
        self.camera_layout = camera_layout
        self.lidar_layout = lidar_layout
        self.patch_size = patch_size

        self.camera_patch = _patch_encoder(
            in_channels[0], patch_hidden_channels, embed_channels, patch_size
        )
        self.lidar_patch = _patch_encoder(
            in_channels[1], patch_hidden_channels, embed_channels, patch_size
        )
        self.depth_attention = DepthGuidedDeformableAttention2D(
            channels=embed_channels,
            num_heads=num_heads,
            downsample_factor=attention_downsample,
            offset_kernel_size=attention_downsample + 2,
            offset_scale=float(attention_downsample),
            dropout=dropout,
        )
        self.attention_to_dense = nn.Sequential(
            nn.Conv2d(embed_channels, embed_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(embed_channels, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                embed_channels,
                out_channels,
                kernel_size=patch_size,
                stride=patch_size,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.camera_residual = nn.Sequential(
            nn.Conv2d(in_channels[0], out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.lidar_residual = nn.Sequential(
            nn.Conv2d(in_channels[1], out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.add_norm = nn.BatchNorm2d(out_channels)
        self.feed_forward = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.ffn_norm = nn.BatchNorm2d(out_channels)

        patch_height = self.grid.height // patch_size
        patch_width = self.grid.width // patch_size
        y, x = torch.meshgrid(
            torch.arange(1, patch_height + 1, dtype=torch.float32),
            torch.arange(1, patch_width + 1, dtype=torch.float32),
            indexing="ij",
        )
        center_x = (patch_width + 1) / 2.0
        center_y = (patch_height + 1) / 2.0
        radius = torch.sqrt((x - center_x).square() + (y - center_y).square())
        edge_x = int(patch_width * 0.9)
        edge_y = int(patch_height * 0.9)
        threshold = ((edge_x - center_x) ** 2 + (edge_y - center_y) ** 2) ** 0.5
        radius = torch.clamp(radius, min=threshold)
        depth = (radius - radius.min()) / (radius.max() - radius.min()) + 1.0
        self.register_buffer("depth_encoding", depth[None, None])

    def forward(self, camera_bev, lidar_bev=None):
        if lidar_bev is None:
            if not isinstance(camera_bev, (list, tuple)) or len(camera_bev) != 2:
                raise ValueError("expected [camera_bev, lidar_bev] inputs")
            camera_bev, lidar_bev = camera_bev
        camera_bev = self.grid.canonicalize(camera_bev, self.camera_layout)
        lidar_bev = self.grid.canonicalize(lidar_bev, self.lidar_layout)
        if camera_bev.shape[0] != lidar_bev.shape[0]:
            raise ValueError("camera and LiDAR batch sizes differ")

        camera_patch = self.camera_patch(camera_bev)
        lidar_patch = self.lidar_patch(lidar_bev)
        depth = self.depth_encoding.to(lidar_patch).expand(
            lidar_patch.shape[0], lidar_patch.shape[1], -1, -1
        )
        attended = self.depth_attention(lidar_patch, camera_patch, depth)
        attended = self.attention_to_dense(attended)
        fused = self.add_norm(
            attended
            + self.lidar_residual(lidar_bev)
            + self.camera_residual(camera_bev)
        )
        return self.ffn_norm(self.feed_forward(fused) + fused)


__all__ = ["DepthGFusion"]
