"""Pure-PyTorch DSVT-Pillar-D128 LiDAR encoder.

The attention, rotated-set partitioning, and BEV backbone follow the official
Apache-2.0 DSVT implementation (https://github.com/Haiyang-W/DSVT).  This
module intentionally implements only the fixed, one-stage nuScenes variant:

    list[Tensor[N_i, 5]] -> Tensor[B, 256, 180, 180]

Topology construction uses PyTorch operations with the same semantics as the
official ``ingroup_inds`` CUDA op.  It is a correctness path, not a claim
about the eventual TensorRT/Thor topology-construction latency.
"""

from typing import List, Sequence, Tuple

import torch
from torch import Tensor, nn


try:
    import torch_scatter
except ImportError:
    torch_scatter = None

def scatter_mean(src: Tensor, index: Tensor, groups: int) -> Tensor:
    if torch_scatter is not None:
        return torch_scatter.scatter_mean(src, index, dim=0, dim_size=groups)
    output = src.new_zeros((groups, src.shape[1]))
    output.index_add_(0, index, src)
    counts = torch.bincount(index, minlength=groups).to(src.dtype).clamp_min_(1)
    return output / counts[:, None]


def scatter_max(src: Tensor, index: Tensor, groups: int) -> Tensor:
    if torch_scatter is not None:
        return torch_scatter.scatter_max(src, index, dim=0, dim_size=groups)[0]
    if not hasattr(Tensor, "scatter_reduce_"):
        raise RuntimeError(
            "DynamicPillarVFE needs torch_scatter or PyTorch >= 1.12"
        )
    output = src.new_full((groups, src.shape[1]), float("-inf"))
    expanded = index[:, None].expand(-1, src.shape[1])
    return output.scatter_reduce_(
        0, expanded, src, reduce="amax", include_self=True
    )


def ingroup_indices(group_ids: Tensor) -> Tensor:
    """Return each element's zero-based rank inside its group."""

    if group_ids.numel() == 0:
        return group_ids.clone()
    order = torch.argsort(group_ids)
    sorted_groups = group_ids[order]
    counts = torch.bincount(sorted_groups)
    starts = torch.cumsum(counts, 0) - counts
    sorted_ranks = torch.arange(
        group_ids.numel(), device=group_ids.device, dtype=group_ids.dtype
    ) - torch.repeat_interleave(starts, counts)
    ranks = torch.empty_like(group_ids)
    ranks[order] = sorted_ranks
    return ranks


def continuous_set_indices(counts: Tensor) -> Tuple[Tensor, Tensor]:
    total = int(counts.sum().item())
    window_indices = torch.repeat_interleave(
        torch.arange(counts.numel(), device=counts.device), counts
    )
    starts = torch.cumsum(counts, 0) - counts
    set_indices = torch.arange(total, device=counts.device) - torch.repeat_interleave(
        starts, counts
    )
    return window_indices, set_indices


def last_occurrence_gather(indices: Tensor) -> Tensor:
    """Map each pillar to its last occurrence in a padded DSVT set tensor."""

    flattened = indices.reshape(-1)
    unique, inverse = torch.unique(flattened, return_inverse=True)
    positions = torch.arange(
        inverse.numel(), device=inverse.device, dtype=inverse.dtype
    )
    output = inverse.new_full((unique.numel(),), -1)
    if torch_scatter is not None:
        return torch_scatter.scatter_max(
            positions, inverse, dim=0, out=output
        )[0]
    if hasattr(output, "scatter_reduce_"):
        return output.scatter_reduce_(
            0, inverse, positions, reduce="amax", include_self=True
        )
    raise RuntimeError(
        "deterministic DSVT gather needs torch_scatter or PyTorch >= 1.12"
    )


def official_occurrence_gather(indices: Tensor) -> Tensor:
    """Reproduce the official reverse/scatter duplicate-index selection."""

    flattened = indices.reshape(-1)
    unique, inverse = torch.unique(flattened, return_inverse=True)
    positions = torch.arange(
        inverse.numel(), device=inverse.device, dtype=inverse.dtype
    )
    inverse = inverse.flip((0,))
    positions = positions.flip((0,))
    return inverse.new_empty(unique.numel()).scatter_(0, inverse, positions)


class DynamicPFNLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, last: bool) -> None:
        super().__init__()
        self.last = last
        units = out_channels if last else out_channels // 2
        self.linear = nn.Linear(in_channels, units, bias=False)
        self.norm = nn.BatchNorm1d(units, eps=1e-3, momentum=0.01)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, features: Tensor, inverse: Tensor, groups: int) -> Tensor:
        features = self.activation(self.norm(self.linear(features)))
        pooled = scatter_max(features, inverse, groups)
        if self.last:
            return pooled
        return torch.cat((features, pooled[inverse]), dim=1)


class DynamicPillarVFE(nn.Module):
    """Dynamic full-height pillarisation plus the official two-layer PFN."""

    def __init__(
        self,
        in_channels: int = 5,
        channels: int = 128,
        voxel_size: Sequence[float] = (0.3, 0.3, 8.0),
        point_cloud_range: Sequence[float] = (-54, -54, -5, 54, 54, 3),
        zero_feature_channels: List[int] = [],
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        if any(
            not isinstance(channel, int) or channel < 0 or channel >= in_channels
            for channel in zero_feature_channels
        ):
            raise ValueError(
                "zero_feature_channels must contain valid zero-based point feature indices"
            )
        self.zero_feature_channels = tuple(zero_feature_channels)
        self.register_buffer("voxel_size", torch.tensor(voxel_size, dtype=torch.float32))
        self.register_buffer(
            "point_cloud_range", torch.tensor(point_cloud_range, dtype=torch.float32)
        )
        grid_size = torch.round(
            (self.point_cloud_range[3:] - self.point_cloud_range[:3]) / self.voxel_size
        ).long()
        self.register_buffer("grid_size", grid_size)
        if grid_size.tolist() != [360, 360, 1]:
            raise ValueError(f"DSVT-Pillar integration requires grid [360,360,1], got {grid_size.tolist()}")

        # Five point channels + xyz cluster offset + xyz pillar-center offset.
        self.layers = nn.ModuleList(
            (
                DynamicPFNLayer(in_channels + 6, channels, last=False),
                DynamicPFNLayer(channels, channels, last=True),
            )
        )

    def forward(self, batches: List[Tensor]) -> Tuple[Tensor, Tensor, int]:
        if not isinstance(batches, (list, tuple)) or not batches:
            raise ValueError("DSVT expects a non-empty list of per-sample point tensors")
        combined = []
        for batch_id, points in enumerate(batches):
            if points.ndim != 2 or points.shape[1] != self.in_channels:
                raise ValueError(
                    f"expected [N,{self.in_channels}] points, got {tuple(points.shape)}"
                )
            combined.append(
                torch.cat((points.new_full((points.shape[0], 1), batch_id), points), dim=1)
            )
        points = torch.cat(combined, dim=0)
        if points.numel() == 0:
            raise ValueError("the complete LiDAR batch contains no points")

        voxel_size = self.voxel_size.to(points.dtype)
        pc_range = self.point_cloud_range.to(points.dtype)
        xy = torch.floor((points[:, 1:3] - pc_range[:2]) / voxel_size[:2]).long()
        valid_xy = ((xy >= 0) & (xy < self.grid_size[:2])).all(dim=1)
        valid_z = (points[:, 3] >= pc_range[2]) & (points[:, 3] < pc_range[5])
        valid = valid_xy & valid_z
        points, xy = points[valid], xy[valid]
        if points.numel() == 0:
            raise ValueError("all LiDAR points are outside point_cloud_range")

        grid_x, grid_y = int(self.grid_size[0]), int(self.grid_size[1])
        merged = points[:, 0].long() * grid_x * grid_y + xy[:, 0] * grid_y + xy[:, 1]
        unique, inverse = torch.unique(merged, sorted=True, return_inverse=True)
        groups = unique.numel()
        xyz = points[:, 1:4].contiguous()
        cluster_offset = xyz - scatter_mean(xyz, inverse, groups)[inverse]
        center_offset = torch.empty_like(xyz)
        center_offset[:, 0] = xyz[:, 0] - (
            xy[:, 0].to(xyz.dtype) * voxel_size[0] + voxel_size[0] / 2 + pc_range[0]
        )
        center_offset[:, 1] = xyz[:, 1] - (
            xy[:, 1].to(xyz.dtype) * voxel_size[1] + voxel_size[1] / 2 + pc_range[1]
        )
        center_offset[:, 2] = xyz[:, 2] - (voxel_size[2] / 2 + pc_range[2])
        point_features = points[:, 1:]
        if self.zero_feature_channels:
            point_features = point_features.clone()
            point_features[:, self.zero_feature_channels] = 0
        features = torch.cat((point_features, cluster_offset, center_offset), dim=1)
        for layer in self.layers:
            features = layer(features, inverse, groups)

        batch = torch.div(unique, grid_x * grid_y, rounding_mode="floor")
        remainder = unique % (grid_x * grid_y)
        x = torch.div(remainder, grid_y, rounding_mode="floor")
        y = remainder % grid_y
        z = torch.zeros_like(batch)
        coords = torch.stack((batch, z, y, x), dim=1).int()
        return features, coords, len(batches)


class PositionEmbedding(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
        )

    def forward(self, coordinates: Tensor) -> Tensor:
        return self.layers(coordinates)


def window_coordinates(
    coords: Tensor,
    sparse_shape: Sequence[int],
    window_shape: Sequence[int],
    shift: Sequence[int],
) -> Tuple[Tensor, Tensor]:
    sparse_x, sparse_y, sparse_z = sparse_shape
    win_x, win_y, win_z = window_shape
    shift_x, shift_y, shift_z = shift
    if sparse_z == win_z:
        shift_z = 0
    shifted_x = coords[:, 3] + shift_x
    shifted_y = coords[:, 2] + shift_y
    shifted_z = coords[:, 1] + shift_z
    win_coord_x = torch.div(shifted_x, win_x, rounding_mode="floor")
    win_coord_y = torch.div(shifted_y, win_y, rounding_mode="floor")
    win_coord_z = torch.div(shifted_z, win_z, rounding_mode="floor")
    max_win_x = (sparse_x + win_x - 1) // win_x + 1
    max_win_y = (sparse_y + win_y - 1) // win_y + 1
    max_win_z = (sparse_z + win_z - 1) // win_z + 1
    windows_per_sample = max_win_x * max_win_y * max_win_z
    window_ids = coords[:, 0] * windows_per_sample
    window_ids += win_coord_x * max_win_y * max_win_z + win_coord_y * max_win_z + win_coord_z
    in_window = torch.stack(
        (shifted_z % win_z, shifted_y % win_y, shifted_x % win_x), dim=1
    )
    return window_ids.long(), in_window.long()


class DSVTInputLayer(nn.Module):
    """Official rotated-set topology for a one-stage sparse pillar map."""

    def __init__(
        self,
        channels: int = 128,
        sparse_shape: Sequence[int] = (360, 360, 1),
        window_shape: Sequence[int] = (30, 30, 1),
        set_size: int = 90,
        block_count: int = 4,
    ) -> None:
        super().__init__()
        self.sparse_shape = tuple(sparse_shape)
        self.window_shape = tuple(window_shape)
        self.shifts = ((0, 0, 0), (window_shape[0] // 2, window_shape[1] // 2, 0))
        self.set_size = set_size
        self.block_count = block_count
        self.position_layers = nn.ModuleList(
            nn.ModuleList((PositionEmbedding(channels), PositionEmbedding(channels)))
            for _ in range(block_count)
        )

    @torch.no_grad()
    def partition_sets(self, window_ids: Tensor, in_window: Tensor) -> Tensor:
        contiguous = torch.unique(window_ids, return_inverse=True)[1]
        voxel_counts = torch.bincount(contiguous)
        set_counts = torch.ceil(voxel_counts / self.set_size).long()
        set_windows, set_in_windows = continuous_set_indices(set_counts)
        win_x, win_y, win_z = self.window_shape
        max_voxels = win_x * win_y * win_z
        selection = set_in_windows[:, None] * self.set_size
        selection = selection + torch.arange(
            self.set_size, device=window_ids.device
        )[None]
        selection = selection * voxel_counts[set_windows, None]
        selection = torch.floor(
            selection.double() / (set_counts[set_windows, None] * self.set_size).double()
        ).long()
        selection += set_windows[:, None] * max_voxels

        inner = ingroup_indices(contiguous)
        _, window_order = torch.sort(contiguous * max_voxels + inner)
        partitions = []
        # First sort y-major, then x-major, exactly as the official input layer.
        for axis in ("y", "x"):
            if axis == "y":
                key = contiguous * max_voxels + in_window[:, 1] * win_x * win_z
                key += in_window[:, 2] * win_z + in_window[:, 0]
            else:
                key = contiguous * max_voxels + in_window[:, 2] * win_y * win_z
                key += in_window[:, 1] * win_z + in_window[:, 0]
            _, axis_order = torch.sort(key)
            axis_inner = torch.empty_like(inner)
            axis_inner.scatter_(0, axis_order, inner[window_order])
            padded_positions = axis_inner + contiguous * max_voxels
            padded = torch.full(
                (voxel_counts.numel() * max_voxels,),
                -1,
                dtype=torch.long,
                device=window_ids.device,
            )
            padded[padded_positions] = torch.arange(
                padded_positions.numel(), device=window_ids.device
            )
            partitions.append(padded[selection])
        return torch.stack(partitions, dim=0)

    def position_embedding(self, coords: Tensor, block: int, shift: int) -> Tensor:
        x = coords[:, 2].float() - self.window_shape[0] / 2
        y = coords[:, 1].float() - self.window_shape[1] / 2
        return self.position_layers[block][shift](torch.stack((x, y), dim=1))

    def forward(self, features: Tensor, coords: Tensor):
        set_indices, set_masks = [], []
        positions = [[] for _ in range(self.block_count)]
        for shift_id, shift in enumerate(self.shifts):
            window_ids, in_window = window_coordinates(
                coords.long(), self.sparse_shape, self.window_shape, shift
            )
            indices = self.partition_sets(window_ids, in_window)
            previous = torch.roll(indices, 1, -1)
            previous[:, :, 0] = -1
            set_indices.append(indices)
            set_masks.append(indices == previous)
            for block in range(self.block_count):
                positions[block].append(self.position_embedding(in_window, block, shift_id))
        return features, set_indices, set_masks, positions


class SetAttention(nn.Module):
    def __init__(
        self,
        channels: int = 128,
        heads: int = 8,
        feedforward: int = 256,
        official_layout: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.official_layout = official_layout
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.linear1 = nn.Linear(channels, feedforward)
        self.linear2 = nn.Linear(feedforward, channels)
        self.activation = nn.GELU()
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

    def forward_with_gather(
        self,
        features: Tensor,
        indices: Tensor,
        mask: Tensor,
        pos: Tensor,
        gather: Tensor,
    ) -> Tensor:
        """Run attention with a topology gather precomputed outside TensorRT."""

        set_features = features[indices]
        set_pos = pos[indices]
        if torch.onnx.is_in_onnx_export():
            attended = self._export_attention(set_features, set_pos, mask)
        else:
            attended = self.attention(
                set_features + set_pos,
                set_features + set_pos,
                set_features,
                key_padding_mask=mask,
                need_weights=False,
            )[0]
        attended = attended.reshape(-1, self.channels)[gather]
        if (
            not torch.onnx.is_in_onnx_export()
            and attended.shape != features.shape
        ):
            raise RuntimeError("DSVT set partition failed to map every pillar exactly once")
        features = self.norm1(features + attended)
        return self.norm2(features + self.linear2(self.activation(self.linear1(features))))

    def _export_attention(
        self, set_features: Tensor, set_pos: Tensor, mask: Tensor
    ) -> Tensor:
        """Export MHA without the bool casts emitted for key_padding_mask."""

        query_key = set_features + set_pos
        weight_q, weight_k, weight_v = self.attention.in_proj_weight.chunk(3)
        if self.attention.in_proj_bias is None:
            bias_q = bias_k = bias_v = None
        else:
            bias_q, bias_k, bias_v = self.attention.in_proj_bias.chunk(3)
        query = nn.functional.linear(query_key, weight_q, bias_q)
        key = nn.functional.linear(query_key, weight_k, bias_k)
        value = nn.functional.linear(set_features, weight_v, bias_v)

        batch, tokens, _ = query.shape
        heads = self.attention.num_heads
        head_channels = self.channels // heads
        query = query.view(batch, tokens, heads, head_channels).transpose(1, 2)
        key = key.view(batch, tokens, heads, head_channels).transpose(1, 2)
        value = value.view(batch, tokens, heads, head_channels).transpose(1, 2)
        query = query * (head_channels ** -0.5)
        scores = torch.matmul(query, key.transpose(-2, -1))
        # Arithmetic masking instead of where/-inf: avoids Cast(to=BOOL) (unsupported
        # by TensorRT 8.5) and the constant-folded full_like tensor (1 GB ONNX bloat).
        # mask is int32 {0,1}; Cast int->float is supported by TensorRT 8.5.
        padding = mask[:, None, None, :].to(scores.dtype)
        scores = scores + padding * -10000.0
        probabilities = torch.softmax(scores, dim=-1)
        attended = torch.matmul(probabilities, value)
        attended = attended.transpose(1, 2).contiguous().view(
            batch, tokens, self.channels
        )
        return self.attention.out_proj(attended)

    def forward(self, features: Tensor, indices: Tensor, mask: Tensor, pos: Tensor) -> Tensor:
        gather = (
            official_occurrence_gather(indices)
            if self.official_layout
            else last_occurrence_gather(indices)
        )
        return self.forward_with_gather(
            features, indices, mask, pos, gather
        )


class DSVTBlock(nn.Module):
    def __init__(self, channels: int = 128, official_layout: bool = True) -> None:
        super().__init__()
        self.official_layout = official_layout
        self.layers = nn.ModuleList(
            (
                SetAttention(channels, official_layout=official_layout),
                SetAttention(channels, official_layout=official_layout),
            )
        )
        if official_layout:
            self.layer_norms = nn.ModuleList(
                (nn.LayerNorm(channels), nn.LayerNorm(channels))
            )

    def forward(self, features, set_indices, set_masks, positions, block_id):
        shift = block_id % 2
        for axis, layer in enumerate(self.layers):
            output = layer(
                features,
                set_indices[shift][axis],
                set_masks[shift][axis],
                positions[axis],
            )
            if self.official_layout:
                output = self.layer_norms[axis](output + features)
            features = output
        return features


class DSVTBackbone(nn.Module):
    def __init__(
        self,
        channels: int = 128,
        set_size: int = 90,
        block_count: int = 4,
        window_shape: Sequence[int] = (30, 30, 1),
        official_layout: bool = True,
    ) -> None:
        super().__init__()
        self.official_layout = official_layout
        self.input_layer = DSVTInputLayer(
            channels, (360, 360, 1), window_shape, set_size, block_count
        )
        self.blocks = nn.ModuleList(
            DSVTBlock(channels, official_layout) for _ in range(block_count)
        )
        self.residual_norms = nn.ModuleList(nn.LayerNorm(channels) for _ in range(block_count))
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, features: Tensor, coords: Tensor) -> Tensor:
        features, indices, masks, positions = self.input_layer(features, coords)
        for block_id, (block, norm) in enumerate(zip(self.blocks, self.residual_norms)):
            residual = features
            features = norm(block(features, indices, masks, positions[block_id], block_id) + residual)
        return features


class DensePillarScatter(nn.Module):
    def __init__(self, channels: int = 128) -> None:
        super().__init__()
        self.channels = channels

    def forward(self, features: Tensor, coords: Tensor, batch_size: int) -> Tensor:
        linear = coords[:, 0].long() * 360 * 360
        linear += coords[:, 2].long() * 360 + coords[:, 3].long()
        canvas = features.new_zeros((batch_size * 360 * 360, self.channels))
        canvas = canvas.index_copy(0, linear, features)
        return canvas.view(batch_size, 360, 360, self.channels).permute(0, 3, 1, 2)


class DSVTBEVNeck(nn.Module):
    """Official DSVT BEV backbone followed by a 384-to-256 adapter."""

    def __init__(self, in_channels: int = 128, out_channels: int = 256) -> None:
        super().__init__()
        layer_nums, strides = (1, 2, 2), (1, 2, 2)
        stage_channels = (128, 128, 256)
        inputs = (in_channels, *stage_channels[:-1])
        self.blocks, self.deblocks = nn.ModuleList(), nn.ModuleList()
        for index in range(3):
            layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(inputs[index], stage_channels[index], 3, stride=strides[index], bias=False),
                nn.BatchNorm2d(stage_channels[index], eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            ]
            for _ in range(layer_nums[index]):
                layers.extend(
                    (
                        nn.Conv2d(stage_channels[index], stage_channels[index], 3, padding=1, bias=False),
                        nn.BatchNorm2d(stage_channels[index], eps=1e-3, momentum=0.01),
                        nn.ReLU(inplace=True),
                    )
                )
            self.blocks.append(nn.Sequential(*layers))
        self.deblocks.extend(
            (
                self.projection(nn.Conv2d(128, 128, 2, stride=2, bias=False)),
                self.projection(nn.ConvTranspose2d(128, 128, 1, stride=1, bias=False)),
                self.projection(nn.ConvTranspose2d(256, 128, 2, stride=2, bias=False)),
            )
        )
        self.adapter = nn.Sequential(
            nn.Conv2d(384, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def projection(layer: nn.Module) -> nn.Sequential:
        return nn.Sequential(
            layer,
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: Tensor) -> Tensor:
        outputs = []
        for block, deblock in zip(self.blocks, self.deblocks):
            features = block(features)
            outputs.append(deblock(features))
        return self.adapter(torch.cat(outputs, dim=1))


class DSVTBEVBasicBlock(nn.Module):
    """BasicBlock used by the official DSVT nuScenes BEV backbone."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)
        self.relu2 = nn.ReLU()
        self.downsample = downsample
        if downsample:
            self.downsample_layer = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, 1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
            )

    def forward(self, features: Tensor) -> Tensor:
        identity = features
        output = self.relu1(self.bn1(self.conv1(features)))
        output = self.bn2(self.conv2(output))
        if self.downsample:
            identity = self.downsample_layer(features)
        output += identity
        return self.relu2(output)


class DSVTBEVResNeck(nn.Module):
    """Official residual BEV backbone followed by a 384-to-256 adapter."""

    def __init__(self, in_channels: int = 128, out_channels: int = 256) -> None:
        super().__init__()
        layer_nums, strides = (1, 2, 2), (1, 2, 2)
        stage_channels = (128, 128, 256)
        inputs = (in_channels, *stage_channels[:-1])
        self.blocks, self.deblocks = nn.ModuleList(), nn.ModuleList()
        for index in range(3):
            layers = [
                DSVTBEVBasicBlock(
                    inputs[index],
                    stage_channels[index],
                    stride=strides[index],
                    downsample=True,
                )
            ]
            layers.extend(
                DSVTBEVBasicBlock(stage_channels[index], stage_channels[index])
                for _ in range(layer_nums[index])
            )
            self.blocks.append(nn.Sequential(*layers))
        self.deblocks.extend(
            (
                self.projection(nn.Conv2d(128, 128, 2, stride=2, bias=False)),
                self.projection(
                    nn.ConvTranspose2d(128, 128, 1, stride=1, bias=False)
                ),
                self.projection(
                    nn.ConvTranspose2d(256, 128, 2, stride=2, bias=False)
                ),
            )
        )
        self.adapter = nn.Sequential(
            nn.Conv2d(384, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def projection(layer: nn.Module) -> nn.Sequential:
        return nn.Sequential(
            layer,
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

    def forward_features(self, features: Tensor) -> Tensor:
        outputs = []
        for block, deblock in zip(self.blocks, self.deblocks):
            features = block(features)
            outputs.append(deblock(features))
        return torch.cat(outputs, dim=1)

    def forward(self, features: Tensor) -> Tensor:
        return self.adapter(self.forward_features(features))


class DSVTLidarEncoder(nn.Module):
    """Raw nuScenes points to the fixed fusion-ready LiDAR BEV interface."""

    bev_layout = "yx"

    def __init__(
        self,
        in_channels: int = 5,
        point_cloud_range: Sequence[float] = (-54, -54, -5, 54, 54, 3),
        voxel_size: Sequence[float] = (0.3, 0.3, 8.0),
        d_model: int = 128,
        set_size: int = 90,
        block_count: int = 4,
        window_shape: Sequence[int] = (30, 30, 1),
        out_channels: int = 256,
        official_layout: bool = True,
        zero_feature_channels: List[int] = [],
    ) -> None:
        super().__init__()
        if d_model != 128 or out_channels != 256:
            raise ValueError("the first integration fixes d_model=128 and out_channels=256")
        self.official_layout = official_layout
        self.vfe = DynamicPillarVFE(
            in_channels,
            d_model,
            voxel_size,
            point_cloud_range,
            zero_feature_channels,
        )
        self.backbone = DSVTBackbone(
            d_model, set_size, block_count, window_shape, official_layout
        )
        self.scatter = DensePillarScatter(d_model)
        neck_type = DSVTBEVResNeck if official_layout else DSVTBEVNeck
        self.neck = neck_type(d_model, out_channels)

    def forward(self, point_batches: List[Tensor]) -> Tensor:
        features, coords, batch_size = self.vfe(point_batches)
        features = self.backbone(features, coords)
        output = self.neck(self.scatter(features, coords, batch_size))
        expected = (batch_size, 256, 180, 180)
        if tuple(output.shape) != expected:
            raise RuntimeError(f"expected LiDAR BEV {expected}, got {tuple(output.shape)}")
        return output
