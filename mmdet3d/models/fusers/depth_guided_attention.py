"""Pure-PyTorch depth-guided deformable attention used by Depth-GFusion."""

from torch import nn
from torch.nn import functional as F
import torch


class ContinuousPositionBias(nn.Module):
    """SwinV2-style continuous relative position bias."""

    def __init__(self, hidden_channels: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, query_grid, sampled_grid, batch_size):
        query_grid = query_grid.reshape(1, -1, 2)
        sampled_grid = sampled_grid.reshape(
            batch_size * self.num_heads, -1, 2
        )
        relative = query_grid[:, :, None] - sampled_grid[:, None]
        relative = torch.sign(relative) * torch.log(relative.abs() + 1.0)
        bias = self.mlp(relative).squeeze(-1)
        return bias.reshape(
            batch_size, self.num_heads, query_grid.shape[1], sampled_grid.shape[1]
        )


class DepthGuidedDeformableAttention2D(nn.Module):
    """DepthFusion-style 2D deformable cross-attention.

    The LiDAR query is multiplicatively modulated by a radial depth encoding.
    Each attention head predicts its own sparse sampling grid over camera BEV,
    which keeps global patch attention substantially smaller than dense MHA.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        downsample_factor: int = 4,
        offset_kernel_size: int = 6,
        offset_scale: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        if (offset_kernel_size - downsample_factor) % 2:
            raise ValueError("offset kernel/stride difference must be even")
        self.channels = channels
        self.num_heads = num_heads
        self.head_channels = channels // num_heads
        self.scale = self.head_channels ** -0.5
        padding = (offset_kernel_size - downsample_factor) // 2

        self.to_query = nn.Conv2d(
            channels, channels, 1, groups=num_heads, bias=False
        )
        self.to_depth = nn.Conv2d(
            channels, channels, 1, groups=num_heads, bias=False
        )
        self.to_offsets = nn.Sequential(
            nn.Conv2d(
                self.head_channels,
                self.head_channels,
                offset_kernel_size,
                stride=downsample_factor,
                padding=padding,
                groups=self.head_channels,
            ),
            nn.GELU(),
            nn.Conv2d(self.head_channels, 2, 1, bias=False),
            nn.Tanh(),
        )
        self.offset_scale = offset_scale
        self.to_key = nn.Conv2d(
            channels, channels, 1, groups=num_heads, bias=False
        )
        self.to_value = nn.Conv2d(
            channels, channels, 1, groups=num_heads, bias=False
        )
        self.position_bias = ContinuousPositionBias(
            max(channels // 4, 16), num_heads
        )
        self.dropout = nn.Dropout(dropout)
        self.to_output = nn.Conv2d(channels, channels, 1)

    @staticmethod
    def _normalized_grid(height, width, device, dtype):
        y, x = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        x = 2.0 * x / max(width - 1, 1) - 1.0
        y = 2.0 * y / max(height - 1, 1) - 1.0
        return torch.stack((x, y), dim=-1)

    def forward(self, lidar_query, camera_key_value, depth_encoding):
        batch, channels, height, width = lidar_query.shape
        if camera_key_value.shape != lidar_query.shape:
            raise ValueError(
                "query/key-value shapes must match, got "
                f"{tuple(lidar_query.shape)} and {tuple(camera_key_value.shape)}"
            )
        if depth_encoding.shape != lidar_query.shape:
            depth_encoding = depth_encoding.expand_as(lidar_query)

        query = self.to_query(lidar_query) * self.to_depth(depth_encoding)
        grouped_query = query.reshape(
            batch * self.num_heads, self.head_channels, height, width
        )
        offsets = self.to_offsets(grouped_query) * self.offset_scale
        sampled_height, sampled_width = offsets.shape[-2:]
        base = self._normalized_grid(
            sampled_height, sampled_width, offsets.device, offsets.dtype
        )
        offset_x = 2.0 * offsets[:, 0] / max(sampled_width - 1, 1)
        offset_y = 2.0 * offsets[:, 1] / max(sampled_height - 1, 1)
        sampled_grid = base[None] + torch.stack((offset_x, offset_y), dim=-1)

        grouped_camera = camera_key_value.reshape(
            batch * self.num_heads, self.head_channels, height, width
        )
        sampled = F.grid_sample(
            grouped_camera,
            sampled_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled.reshape(
            batch, channels, sampled_height, sampled_width
        )
        key = self.to_key(sampled)
        value = self.to_value(sampled)

        query = query.reshape(
            batch, self.num_heads, self.head_channels, height * width
        ).permute(0, 1, 3, 2)
        key = key.reshape(
            batch,
            self.num_heads,
            self.head_channels,
            sampled_height * sampled_width,
        ).permute(0, 1, 3, 2)
        value = value.reshape(
            batch,
            self.num_heads,
            self.head_channels,
            sampled_height * sampled_width,
        ).permute(0, 1, 3, 2)

        similarity = torch.matmul(query * self.scale, key.transpose(-1, -2))
        query_grid = self._normalized_grid(
            height, width, similarity.device, similarity.dtype
        )
        similarity = similarity + self.position_bias(
            query_grid, sampled_grid, batch
        )
        similarity = similarity - similarity.max(
            dim=-1, keepdim=True
        ).values.detach()
        attention = self.dropout(similarity.softmax(dim=-1))
        output = torch.matmul(attention, value)
        output = output.permute(0, 1, 3, 2).reshape(
            batch, channels, height, width
        )
        return self.to_output(output)


__all__ = ["DepthGuidedDeformableAttention2D"]
