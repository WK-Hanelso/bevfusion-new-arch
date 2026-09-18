"""Minimal WidthFormer view transform for the dynamic BEVFusion path.

This module adapts only the view-transform ideas needed from the official
WidthFormer implementation: vertical feature compression, calibration-aware
polar positional embeddings, and one semantic single-head cross-attention
layer.  It intentionally does not import the official BEVDet plugin stack.

Reference:
https://github.com/ChenhongyiYang/WidthFormer
"""

import math
from typing import Sequence, Tuple

import torch
from mmcv.runner import force_fp32
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import VTRANSFORMS

__all__ = ["WidthFormerTransform"]


def _position_embedding(pos: torch.Tensor, channels: int) -> torch.Tensor:
    """Encode one scalar per token into ``channels`` sine/cosine features."""
    if channels % 2:
        raise ValueError(f"position embedding channels must be even, got {channels}")
    dim = torch.arange(channels, dtype=torch.float32, device=pos.device)
    dim = 10000 ** (2 * torch.div(dim, 2, rounding_mode="floor") / channels)
    phase = pos.float().unsqueeze(-1) * (2 * math.pi) / dim
    return torch.stack((phase[..., 0::2].sin(), phase[..., 1::2].cos()), -1).flatten(-2)


def _polar_embedding(x: torch.Tensor, y: torch.Tensor, channels: int) -> torch.Tensor:
    radius = torch.sqrt(x.square() + y.square()).clamp_min(1e-6)
    direction = torch.stack((y / radius, x / radius), dim=-1)
    direction_embedding = torch.cat(
        (_position_embedding(direction[..., 0], channels),
         _position_embedding(direction[..., 1], channels)),
        dim=-1,
    )
    return torch.cat((_position_embedding(radius, channels), direction_embedding), dim=-1)


class _HorizontalRefiner(nn.Module):
    """Compress image height to width tokens, then refine each image column."""

    def __init__(self, in_channels: int, channels: int, ffn_channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.vertical_pe = nn.Sequential(
            nn.Linear(channels, channels), nn.ReLU(), nn.Linear(channels, channels)
        )
        self.horizontal_pe = nn.Sequential(
            nn.Linear(channels, channels), nn.ReLU(), nn.Linear(channels, channels)
        )

        self.cross_q_norm = nn.LayerNorm(channels)
        self.cross_k_norm = nn.LayerNorm(channels)
        self.cross_v_norm = nn.LayerNorm(channels)
        self.cross_q = nn.Linear(channels, channels)
        self.cross_k = nn.Linear(channels, channels, bias=False)
        self.cross_v = nn.Linear(channels, channels)
        self.cross_out = nn.Linear(channels, channels)

        self.self_q_norm = nn.LayerNorm(channels)
        self.self_v_norm = nn.LayerNorm(channels)
        self.self_q = nn.Linear(channels, channels)
        self.self_k = nn.Linear(channels, channels, bias=False)
        self.self_v = nn.Linear(channels, channels)
        self.self_out = nn.Linear(channels, channels)

        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, ffn_channels),
            nn.GELU(),
            nn.Linear(ffn_channels, channels),
        )

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.projection(image)
        batch, channels, height, width = image.shape
        dtype = image.dtype
        h_coord = (torch.arange(height, device=image.device, dtype=torch.float32) + 0.5) / height
        w_coord = (torch.arange(width, device=image.device, dtype=torch.float32) + 0.5) / width
        h_pe = self.vertical_pe(_position_embedding(h_coord * 10.0, channels)).to(dtype)
        w_pe = self.horizontal_pe(_position_embedding(w_coord * 10.0, channels)).to(dtype)

        token = image.max(dim=2).values.transpose(1, 2)
        column = image.permute(0, 3, 2, 1).reshape(batch * width, height, channels)
        query = self.cross_q_norm(token).reshape(batch * width, 1, channels)
        key = self.cross_k_norm(column + h_pe.view(1, height, channels))
        value = self.cross_v_norm(column)
        attention = torch.matmul(
            self.cross_q(query), self.cross_k(key).transpose(-2, -1)
        ) * (channels ** -0.5)
        attention = attention.softmax(dim=-1)
        refined = torch.matmul(attention, self.cross_v(value)).reshape(batch, width, channels)
        token = token + self.cross_out(refined)

        query_key = self.self_q_norm(token + w_pe.view(1, width, channels))
        value = self.self_v_norm(token)
        attention = torch.matmul(
            self.self_q(query_key), self.self_k(query_key).transpose(-2, -1)
        ) * (channels ** -0.5)
        token = token + self.self_out(torch.matmul(attention.softmax(dim=-1), self.self_v(value)))
        token = token + self.ffn(self.ffn_norm(token))
        return token, image


@VTRANSFORMS.register_module()
class WidthFormerTransform(nn.Module):
    """Generate canonical ``[B, C, Y, X]`` BEV features without BEV pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Sequence[float],
        ybound: Sequence[float],
        zbound: Sequence[float],
        dbound: Sequence[float],
        ffn_channels: int = 320,
        attention_chunk_size: int = 0,
        positional_scale: float = 10.0,
    ) -> None:
        super().__init__()
        if out_channels % 2:
            raise ValueError("out_channels must be even for sine/cosine embeddings")
        self.out_channels = out_channels
        self.image_size = tuple(image_size)
        self.feature_size = tuple(feature_size)
        self.xbound = tuple(float(v) for v in xbound)
        self.ybound = tuple(float(v) for v in ybound)
        self.zbound = tuple(float(v) for v in zbound)
        self.dbound = tuple(float(v) for v in dbound)
        self.attention_chunk_size = int(attention_chunk_size)
        self.positional_scale = float(positional_scale)

        self.bev_height = round((self.ybound[1] - self.ybound[0]) / self.ybound[2])
        self.bev_width = round((self.xbound[1] - self.xbound[0]) / self.xbound[2])
        depths = torch.arange(*self.dbound, dtype=torch.float32)
        self.register_buffer("depth_values", depths, persistent=False)

        self.refiner = _HorizontalRefiner(in_channels, out_channels, ffn_channels)
        self.depth_net = nn.Conv2d(out_channels, len(depths), 1)
        self.height_net = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 1, 3, padding=1),
        )
        self.image_position_mlp = nn.Sequential(
            nn.Linear(out_channels * 3, out_channels * 4),
            nn.ReLU(),
            nn.Linear(out_channels * 4, out_channels),
        )
        self.bev_position_mlp = nn.Sequential(
            nn.Linear(out_channels * 3, out_channels * 4),
            nn.ReLU(),
            nn.Linear(out_channels * 4, out_channels),
        )

        self.query_norm = nn.LayerNorm(out_channels)
        self.key_norm = nn.LayerNorm(out_channels)
        self.value_norm = nn.LayerNorm(out_channels)
        self.query_projection = nn.Linear(out_channels, out_channels)
        self.key_projection = nn.Linear(out_channels, out_channels, bias=False)
        self.value_projection = nn.Linear(out_channels, out_channels)
        self.semantic_projection = nn.Linear(out_channels, 1)
        self.output_projection = nn.Linear(out_channels, out_channels)
        self.bev_ffn = nn.Sequential(
            nn.Conv2d(out_channels, ffn_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(ffn_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(ffn_channels, out_channels, 1),
        )
        self.fp16_enabled = False

    def _geometry(
        self,
        camera2lidar: torch.Tensor,
        intrinsics: torch.Tensor,
        image_aug: torch.Tensor,
        lidar_aug: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Return calibrated points as ``[B,N,D,H,W,3]`` in augmented LiDAR."""
        batch, views = camera2lidar.shape[:2]
        image_h, image_w = self.image_size
        xs = torch.linspace(0, image_w - 1, width, device=camera2lidar.device)
        ys = torch.linspace(0, image_h - 1, height, device=camera2lidar.device)
        depth = self.depth_values.to(camera2lidar.device)
        dd, yy, xx = torch.meshgrid(depth, ys, xs, indexing="ij")
        points = torch.stack((xx, yy, dd), dim=-1)
        points = points.view(1, 1, len(depth), height, width, 3)

        post_rot = image_aug[..., :3, :3].float()
        post_trans = image_aug[..., :3, 3].float()
        points = points - post_trans.view(batch, views, 1, 1, 1, 3)
        points = torch.linalg.inv(post_rot).view(batch, views, 1, 1, 1, 3, 3).matmul(
            points.unsqueeze(-1)
        )
        points = torch.cat((points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), dim=-2)

        cam_rot = camera2lidar[..., :3, :3].float()
        cam_trans = camera2lidar[..., :3, 3].float()
        intrinsics = intrinsics[..., :3, :3].float()
        transform = cam_rot.matmul(torch.linalg.inv(intrinsics))
        points = transform.view(batch, views, 1, 1, 1, 3, 3).matmul(points)[..., 0]
        points = points + cam_trans.view(batch, views, 1, 1, 1, 3)

        aug_rot = lidar_aug[..., :3, :3].float()
        aug_trans = lidar_aug[..., :3, 3].float()
        points = aug_rot.view(batch, 1, 1, 1, 1, 3, 3).matmul(
            points.unsqueeze(-1)
        )[..., 0]
        return points + aug_trans.view(batch, 1, 1, 1, 1, 3)

    def _image_position(
        self, geometry: torch.Tensor, depth_probability: torch.Tensor, height_logits: torch.Tensor
    ) -> torch.Tensor:
        x_range = self.xbound[1] - self.xbound[0]
        y_range = self.ybound[1] - self.ybound[0]
        x = ((geometry[..., 0] - self.xbound[0]) / x_range - 0.5) * self.positional_scale
        y = ((geometry[..., 1] - self.ybound[0]) / y_range - 0.5) * self.positional_scale

        # Match WidthFormer RefPE: depth probabilities weight radial encodings,
        # while ray direction is computed once per column from the far endpoint.
        radius = torch.sqrt(x.square() + y.square()).clamp_min(1e-6)
        radius_embedding = _position_embedding(radius, self.out_channels)
        expected_radius = (
            depth_probability.unsqueeze(-1) * radius_embedding
        ).sum(dim=2)
        height_probability = height_logits.softmax(dim=2).unsqueeze(-1)
        expected_radius = (height_probability * expected_radius).sum(dim=2)

        far_x = x[:, :, -1, 0]
        far_y = y[:, :, -1, 0]
        far_radius = torch.sqrt(far_x.square() + far_y.square()).clamp_min(1e-6)
        direction_embedding = torch.cat(
            (
                _position_embedding(far_y / far_radius, self.out_channels),
                _position_embedding(far_x / far_radius, self.out_channels),
            ),
            dim=-1,
        )
        position = torch.cat((expected_radius, direction_embedding), dim=-1)
        return self.image_position_mlp(position)

    def _bev_position(self, batch: int, device: torch.device) -> torch.Tensor:
        x = (torch.arange(self.bev_width, device=device, dtype=torch.float32) + 0.5) / self.bev_width - 0.5
        y = (torch.arange(self.bev_height, device=device, dtype=torch.float32) + 0.5) / self.bev_height - 0.5
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        position = _polar_embedding(
            xx * self.positional_scale,
            yy * self.positional_scale,
            self.out_channels,
        )
        position = self.bev_position_mlp(position)
        return position.view(1, self.bev_height * self.bev_width, self.out_channels).expand(batch, -1, -1)

    def _semantic_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        query_projected = self.query_projection(self.query_norm(query))
        key_projected = self.key_projection(self.key_norm(key))
        value_normed = self.value_norm(value)
        value_projected = self.value_projection(value_normed)
        semantic = self.semantic_projection(value_normed.detach()).transpose(1, 2)
        scale = self.out_channels ** -0.5

        def attend(query_chunk: torch.Tensor) -> torch.Tensor:
            score = torch.matmul(query_chunk, key_projected.transpose(1, 2)) * scale
            score = score + semantic
            return torch.matmul(score.softmax(dim=-1), value_projected)

        chunk = self.attention_chunk_size
        query_tokens = self.bev_height * self.bev_width
        if chunk > 0 and query_tokens > chunk:
            output = torch.cat(
                [attend(part) for part in query_projected.split(chunk, dim=1)], dim=1
            )
        else:
            output = attend(query_projected)
        return query + self.output_projection(output)

    def forward_with_geometry(
        self, image: torch.Tensor, geometry: torch.Tensor
    ) -> torch.Tensor:
        """Generate BEV from image features and precomputed calibrated geometry.

        This explicit deployment boundary keeps matrix inversion outside the
        TensorRT graph while retaining depth-aware RefPE and BEV attention.
        """
        batch, views, channels, height, width = image.shape
        if (
            not torch.onnx.is_in_onnx_export()
            and (height, width) != self.feature_size
        ):
            raise ValueError(
                f"WidthFormer feature_size={self.feature_size}, got {(height, width)}"
            )
        expected_geometry_shape = (
            batch, views, len(self.depth_values), height, width, 3
        )
        if (
            not torch.onnx.is_in_onnx_export()
            and tuple(geometry.shape) != expected_geometry_shape
        ):
            raise ValueError(
                f"expected geometry {expected_geometry_shape}, got {tuple(geometry.shape)}"
            )
        width_tokens, projected = self.refiner(
            image.reshape(batch * views, channels, height, width)
        )
        width_tokens = width_tokens.view(batch, views, width, self.out_channels)
        projected = projected.view(batch, views, self.out_channels, height, width)

        flat = projected.reshape(batch * views, self.out_channels, height, width)
        depth_probability = self.depth_net(flat).view(
            batch, views, len(self.depth_values), height, width
        ).softmax(dim=2)
        height_logits = self.height_net(flat).view(batch, views, height, width)
        image_position = self._image_position(geometry, depth_probability, height_logits)

        key = (width_tokens + image_position).reshape(batch, views * width, self.out_channels)
        value = width_tokens.reshape(batch, views * width, self.out_channels)
        query = self._bev_position(batch, image.device).to(image.dtype)
        bev = self._semantic_attention(query, key, value)
        bev = bev.transpose(1, 2).reshape(
            batch, self.out_channels, self.bev_height, self.bev_width
        )
        return bev + self.bev_ffn(bev)

    @force_fp32()
    def forward(
        self,
        image: torch.Tensor,
        points,
        radar,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas=None,
        **kwargs,
    ) -> torch.Tensor:
        height, width = image.shape[-2:]
        geometry = self._geometry(
            camera2lidar, camera_intrinsics, img_aug_matrix, lidar_aug_matrix, height, width
        )
        return self.forward_with_geometry(image, geometry)
