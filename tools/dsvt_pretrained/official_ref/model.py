# Copyright 2023 Haiyang-W/DSVT contributors.
# SPDX-License-Identifier: Apache-2.0
"""Faithful, CPU-only vendoring of the official one-stage nuScenes DSVT path.

The module names intentionally match the official checkpoint. Only generic
OpenPCDet configuration plumbing and the CUDA ``ingroup_inds`` op are removed.
See NOTICE.md for the exact upstream source files.
"""

from typing import Dict, List, Sequence, Tuple

import torch
from torch import Tensor, nn


def _scatter_mean(src: Tensor, inverse: Tensor, groups: int) -> Tensor:
    out = src.new_zeros((groups, src.shape[1]))
    out.index_add_(0, inverse, src)
    count = torch.bincount(inverse, minlength=groups).to(src.dtype).clamp_min_(1)
    return out / count[:, None]


def _scatter_max(src: Tensor, inverse: Tensor, groups: int) -> Tensor:
    out = src.new_full((groups, src.shape[1]), float("-inf"))
    return out.scatter_reduce_(
        0,
        inverse[:, None].expand(-1, src.shape[1]),
        src,
        reduce="amax",
        include_self=True,
    )


def ingroup_inds(group_ids: Tensor) -> Tensor:
    """CPU equivalent of the official atomic group occurrence counter."""

    result = torch.empty_like(group_ids)
    counters: Dict[int, int] = {}
    for index, group in enumerate(group_ids.tolist()):
        result[index] = counters.get(group, 0)
        counters[group] = counters.get(group, 0) + 1
    return result


def _continuous_indices(counts: Tensor) -> Tuple[Tensor, Tensor]:
    total = int(counts.sum().item())
    windows = torch.repeat_interleave(
        torch.arange(counts.numel(), device=counts.device), counts
    )
    starts = torch.cumsum(counts, 0) - counts
    within = torch.arange(total, device=counts.device) - torch.repeat_interleave(
        starts, counts
    )
    return windows, within


def _window_coordinates(
    coords: Tensor, window_shape: Sequence[int], shift: Sequence[int]
) -> Tuple[Tensor, Tensor]:
    # Official sparse shape is fixed at [360, 360, 1].
    win_x, win_y, win_z = window_shape
    sx, sy, _ = shift
    shifted_x = coords[:, 3] + sx
    shifted_y = coords[:, 2] + sy
    shifted_z = coords[:, 1]
    wx = torch.div(shifted_x, win_x, rounding_mode="floor")
    wy = torch.div(shifted_y, win_y, rounding_mode="floor")
    wz = torch.div(shifted_z, win_z, rounding_mode="floor")
    # ceil(360 / 30) + 1, and ceil(1 / 1) + 1, from dsvt_utils.py.
    max_x, max_y, max_z = 13, 13, 2
    ids = coords[:, 0] * (max_x * max_y * max_z)
    ids = ids + wx * max_y * max_z + wy * max_z + wz
    within = torch.stack(
        (shifted_z % win_z, shifted_y % win_y, shifted_x % win_x), dim=1
    )
    return ids.long(), within.long()


class PFNLayerV2(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, last_layer: bool):
        super().__init__()
        self.last_vfe = last_layer
        units = out_channels if last_layer else out_channels // 2
        self.linear = nn.Linear(in_channels, units, bias=False)
        self.norm = nn.BatchNorm1d(units, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU()

    def forward(self, inputs: Tensor, inverse: Tensor, groups: int) -> Tensor:
        features = self.relu(self.norm(self.linear(inputs)))
        maximum = _scatter_max(features, inverse, groups)
        if self.last_vfe:
            return maximum
        return torch.cat((features, maximum[inverse]), dim=1)


class DynamicPillarVFE(nn.Module):
    def __init__(self):
        super().__init__()
        self.pfn_layers = nn.ModuleList(
            (PFNLayerV2(11, 128, False), PFNLayerV2(128, 128, True))
        )
        # Plain attributes match upstream: these are not checkpoint buffers.
        self.voxel_size = torch.tensor((0.3, 0.3, 8.0), dtype=torch.float32)
        self.point_cloud_range = torch.tensor(
            (-54.0, -54.0, -5.0, 54.0, 54.0, 3.0), dtype=torch.float32
        )
        self.grid_size = torch.tensor((360, 360, 1), dtype=torch.long)

    def forward(self, points: Tensor) -> Tuple[Tensor, Tensor]:
        # points: [batch, x, y, z, intensity, timestamp]
        coords_xy = torch.floor(
            (points[:, 1:3] - self.point_cloud_range[:2]) / self.voxel_size[:2]
        ).int()
        mask = ((coords_xy >= 0) & (coords_xy < self.grid_size[:2])).all(dim=1)
        points, coords_xy = points[mask], coords_xy[mask]
        xyz = points[:, 1:4].contiguous()
        merged = (
            points[:, 0].int() * (360 * 360)
            + coords_xy[:, 0] * 360
            + coords_xy[:, 1]
        )
        unique, inverse = torch.unique(merged, return_inverse=True)
        groups = unique.numel()
        cluster = xyz - _scatter_mean(xyz, inverse, groups)[inverse]
        center = torch.zeros_like(xyz)
        center[:, 0] = xyz[:, 0] - (coords_xy[:, 0].to(xyz.dtype) * 0.3 - 53.85)
        center[:, 1] = xyz[:, 1] - (coords_xy[:, 1].to(xyz.dtype) * 0.3 - 53.85)
        center[:, 2] = xyz[:, 2] - (-1.0)
        features = torch.cat((points[:, 1:], cluster, center), dim=1)
        for layer in self.pfn_layers:
            features = layer(features, inverse, groups)
        unique = unique.int()
        coords = torch.stack(
            (
                torch.div(unique, 360 * 360, rounding_mode="floor"),
                torch.zeros_like(unique),
                unique % 360,
                torch.div(unique % (360 * 360), 360, rounding_mode="floor"),
            ),
            dim=1,
        )
        return features, coords


class PositionEmbeddingLearned(nn.Module):
    def __init__(self):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Linear(2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
        )

    def forward(self, xy: Tensor) -> Tensor:
        return self.position_embedding_head(xy)


class DSVTInputLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.posembed_layers = nn.ModuleList(
            nn.ModuleList(
                nn.ModuleList((PositionEmbeddingLearned(), PositionEmbeddingLearned()))
                for _ in range(4)
            )
            for _ in range(1)
        )
        self.window_shape = (30, 30, 1)
        self.shifts = ((0, 0, 0), (15, 15, 0))
        self.set_size = 90

    @torch.no_grad()
    def _sets(self, window_ids: Tensor, within: Tensor) -> Tensor:
        contiguous = torch.unique(window_ids, return_inverse=True)[1]
        counts = torch.bincount(contiguous)
        set_counts = torch.ceil(counts / self.set_size).long()
        set_windows, within_sets = _continuous_indices(set_counts)
        max_voxels = 30 * 30
        selection = within_sets[:, None] * self.set_size
        selection = selection + torch.arange(self.set_size, device=window_ids.device)
        selection = selection * counts[set_windows, None]
        selection = torch.floor(
            selection.double()
            / (set_counts[set_windows, None] * self.set_size).double()
        ).long()
        selection = selection + set_windows[:, None] * max_voxels

        inner = ingroup_inds(contiguous)
        _, input_order = torch.sort(contiguous * max_voxels + inner)
        partitions: List[Tensor] = []
        for axis in ("y", "x"):
            if axis == "y":
                key = contiguous * max_voxels + within[:, 1] * 30 + within[:, 2]
            else:
                key = contiguous * max_voxels + within[:, 2] * 30 + within[:, 1]
            _, axis_order = torch.sort(key)
            axis_inner = torch.empty_like(inner)
            axis_inner.scatter_(0, axis_order, inner[input_order])
            positions = axis_inner + contiguous * max_voxels
            padded = torch.full(
                (counts.numel() * max_voxels,),
                -1,
                dtype=torch.long,
                device=window_ids.device,
            )
            padded[positions] = torch.arange(positions.numel(), device=positions.device)
            partitions.append(padded[selection])
        return torch.stack(partitions, dim=0)

    def forward(self, features: Tensor, coords: Tensor):
        indices, masks = [], []
        positions = [[] for _ in range(4)]
        for shift_id, shift in enumerate(self.shifts):
            window_ids, within = _window_coordinates(coords.long(), self.window_shape, shift)
            partition = self._sets(window_ids, within)
            previous = torch.roll(partition, 1, -1)
            previous[:, :, 0] = -1
            indices.append(partition)
            masks.append(partition == previous)
            x = within[:, 2].float() - 15.0
            y = within[:, 1].float() - 15.0
            xy = torch.stack((x, y), dim=1)
            for block_id in range(4):
                positions[block_id].append(
                    self.posembed_layers[0][block_id][shift_id](xy)
                )
        return features.clone(), indices, masks, positions


def _official_gather(indices: Tensor) -> Tensor:
    flattened = indices.reshape(-1)
    unique, inverse = torch.unique(flattened, return_inverse=True)
    positions = torch.arange(inverse.numel(), device=inverse.device, dtype=inverse.dtype)
    inverse, positions = inverse.flip((0,)), positions.flip((0,))
    return inverse.new_empty(unique.numel()).scatter_(0, inverse, positions)


class SetAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.nhead = 8
        self.self_attn = nn.MultiheadAttention(128, 8, dropout=0.0, batch_first=True)
        self.linear1 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(0)
        self.linear2 = nn.Linear(256, 128)
        self.d_model = 128
        self.norm1 = nn.LayerNorm(128)
        self.norm2 = nn.LayerNorm(128)
        self.dropout1 = nn.Identity()
        self.dropout2 = nn.Identity()

    def forward(
        self, src: Tensor, pos: Tensor, mask: Tensor, voxel_inds: Tensor
    ) -> Tensor:
        set_features = src[voxel_inds]
        set_pos = pos[voxel_inds]
        attended = self.self_attn(
            set_features + set_pos,
            set_features + set_pos,
            set_features,
            key_padding_mask=mask,
        )[0]
        attended = attended.reshape(-1, 128)[_official_gather(voxel_inds)]
        src = self.norm1(src + attended)
        return self.norm2(src + self.linear2(self.dropout(torch.nn.functional.gelu(self.linear1(src)))))


class DSVT_EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.win_attn = SetAttention()
        self.norm = nn.LayerNorm(128)
        self.d_model = 128

    def forward(self, src: Tensor, indices: Tensor, mask: Tensor, pos: Tensor) -> Tensor:
        identity = src
        src = self.win_attn(src, pos, mask, indices)
        return self.norm(src + identity)


class DSVTBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_list = nn.ModuleList((DSVT_EncoderLayer(), DSVT_EncoderLayer()))

    def forward(self, src, indices, masks, positions, block_id):
        output = src
        shift = block_id % 2
        for axis, layer in enumerate(self.encoder_list):
            output = layer(
                output, indices[shift][axis], masks[shift][axis], positions[axis]
            )
        return output


class DSVT(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = DSVTInputLayer()
        self.stage_0 = nn.ModuleList(DSVTBlock() for _ in range(4))
        self.residual_norm_stage_0 = nn.ModuleList(nn.LayerNorm(128) for _ in range(4))

    def forward_trace(self, features: Tensor, coords: Tensor) -> Tuple[Tensor, List[Tensor]]:
        output, indices, masks, positions = self.input_layer(features, coords)
        trace = []
        for block_id, (block, norm) in enumerate(
            zip(self.stage_0, self.residual_norm_stage_0)
        ):
            residual = output.clone()
            output = norm(
                block(output, indices, masks, positions[block_id], block_id) + residual
            )
            trace.append(output)
        return output, trace


class PointPillarScatter3d(nn.Module):
    def forward(self, features: Tensor, coords: Tensor, batch_size: int) -> Tensor:
        canvases = []
        for batch_id in range(batch_size):
            canvas = features.new_zeros((128, 360 * 360))
            selected = coords[:, 0] == batch_id
            current = coords[selected]
            indices = current[:, 1] * 360 * 360 + current[:, 2] * 360 + current[:, 3]
            canvas[:, indices.long()] = features[selected].t()
            canvases.append(canvas)
        return torch.stack(canvases).view(batch_size, 128, 360, 360)


class BasicBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.relu2 = nn.ReLU()
        self.downsample = downsample
        if downsample:
            self.downsample_layer = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01),
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        output = self.relu1(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        if self.downsample:
            identity = self.downsample_layer(x)
        return self.relu2(output + identity)


class BaseBEVResBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        layer_nums, strides = (1, 2, 2), (1, 2, 2)
        channels, inputs = (128, 128, 256), (128, 128, 128)
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for index in range(3):
            layers = [BasicBlock(inputs[index], channels[index], strides[index], True)]
            layers.extend(BasicBlock(channels[index], channels[index]) for _ in range(layer_nums[index]))
            self.blocks.append(nn.Sequential(*layers))
        self.deblocks.extend(
            (
                self._projection(nn.Conv2d(128, 128, 2, stride=2, bias=False)),
                self._projection(nn.ConvTranspose2d(128, 128, 1, stride=1, bias=False)),
                self._projection(nn.ConvTranspose2d(256, 128, 2, stride=2, bias=False)),
            )
        )

    @staticmethod
    def _projection(layer: nn.Module) -> nn.Sequential:
        return nn.Sequential(
            layer,
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

    def forward(self, spatial_features: Tensor) -> Tensor:
        outputs = []
        x = spatial_features
        for block, deblock in zip(self.blocks, self.deblocks):
            x = block(x)
            outputs.append(deblock(x))
        return torch.cat(outputs, dim=1)


class OfficialDSVTLidar(nn.Module):
    """Relevant official modules, preserving their checkpoint prefix names."""

    def __init__(self):
        super().__init__()
        self.vfe = DynamicPillarVFE()
        self.backbone_3d = DSVT()
        self.map_to_bev = PointPillarScatter3d()
        self.backbone_2d = BaseBEVResBackbone()

    def forward_trace(self, points: Tensor, batch_size: int = 1):
        vfe, coords = self.vfe(points)
        encoded, blocks = self.backbone_3d.forward_trace(vfe, coords)
        bev = self.map_to_bev(encoded, coords, batch_size)
        output = self.backbone_2d(bev)
        return {
            "coords": coords,
            "vfe": vfe,
            "blocks": blocks,
            "bev": bev,
            "backbone_2d": output,
        }
