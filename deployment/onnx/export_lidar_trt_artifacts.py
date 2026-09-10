#!/usr/bin/env python3
"""Export all learned artifacts required for raw-points TensorRT Engine B."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch import nn

from deployment.onnx import dsvt_backbone as dsvt_export
from deployment.onnx.model_contract import (
    DEFAULT_CONFIG,
    LIDAR_BEV_SHAPE,
    load_deployment_model,
    sha256_file,
)
from deployment.onnx.onnx_compat import load_and_check_onnx, prepare_onnx_export


INPUT_NAMES = (
    "src",
    "set_indices_shift_0",
    "set_indices_shift_1",
    "set_masks_shift_0",
    "set_masks_shift_1",
    "gather_shift_0",
    "gather_shift_1",
    "position_embeddings",
)
GRID_SIZE = (360, 360, 1)
WINDOW_SIZE = (30, 30, 1)
SET_SIZE = 90
MAX_GRID_PILLARS = GRID_SIZE[0] * GRID_SIZE[1]
MAX_GRID_SETS = 13 * 13 * 10


class PaddedDSVTBackbone(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.transformer = dsvt_export.DSVTDeployWrapper(backbone)

    def forward(
        self,
        src,
        indices0,
        indices1,
        masks0,
        masks1,
        gather0,
        gather1,
        positions,
    ):
        return self.transformer(
            src,
            indices0.long(),
            indices1.long(),
            masks0.bool(),
            masks1.bool(),
            gather0.long(),
            gather1.long(),
            positions,
        )


class DSVTBEVNeck(nn.Module):
    def __init__(self, neck):
        super().__init__()
        self.neck = neck

    def forward(self, dense_bev):
        return self.neck(dense_bev)


def validate_encoder_contract(encoder, max_points, max_pillars, max_sets):
    if max_points < 1:
        raise ValueError("--max-points must be positive")
    if not 1 <= max_pillars <= MAX_GRID_PILLARS:
        raise ValueError(f"--max-pillars must be in [1,{MAX_GRID_PILLARS}]")
    if not 1 <= max_sets <= MAX_GRID_SETS:
        raise ValueError(f"--max-sets must be in [1,{MAX_GRID_SETS}]")

    vfe = encoder.vfe
    actual_grid = tuple(int(value) for value in vfe.grid_size.tolist())
    actual_voxel = tuple(float(value) for value in vfe.voxel_size.tolist())
    actual_range = tuple(float(value) for value in vfe.point_cloud_range.tolist())
    input_layer = encoder.backbone.input_layer
    expected = {
        "grid": GRID_SIZE,
        "voxel": (0.3, 0.3, 8.0),
        "range": (-54.0, -54.0, -5.0, 54.0, 54.0, 3.0),
        "window": WINDOW_SIZE,
        "set_size": SET_SIZE,
        "blocks": 4,
    }
    discrete_actual = {
        "grid": actual_grid,
        "window": tuple(input_layer.window_shape),
        "set_size": int(input_layer.set_size),
        "blocks": int(input_layer.block_count),
    }
    for key, value in discrete_actual.items():
        if value != expected[key]:
            raise ValueError(
                f"TensorRT LiDAR plugin contract mismatch for {key}: "
                f"expected={expected[key]}, actual={value}"
            )
    for key, value in (("voxel", actual_voxel), ("range", actual_range)):
        if not np.allclose(value, expected[key], rtol=0.0, atol=1e-6):
            raise ValueError(
                f"TensorRT LiDAR plugin contract mismatch for {key}: "
                f"expected={expected[key]}, actual={value}"
            )


def padded_inputs(backbone, example_pillars, max_pillars, max_sets, device):
    dynamic, _ = dsvt_export.make_inputs(backbone, example_pillars, device)
    src, indices0, indices1, masks0, masks1, gather0, gather1, positions = dynamic
    if example_pillars > max_pillars:
        raise ValueError(
            f"example pillars {example_pillars} exceed max pillars {max_pillars}"
        )

    padded_src = src.new_zeros((max_pillars, 128))
    padded_src[:example_pillars] = src
    output_indices = []
    output_masks = []
    output_gathers = []
    for shift, (indices, masks, gather) in enumerate(
        ((indices0, masks0, gather0), (indices1, masks1, gather1))
    ):
        set_count = int(indices.shape[1])
        if set_count > max_sets:
            raise ValueError(
                f"example shift {shift} set count {set_count} exceeds "
                f"max sets {max_sets}"
            )
        padded_indices = torch.zeros(
            (2, max_sets, SET_SIZE), dtype=torch.int32, device=device
        )
        padded_masks = torch.ones(
            (2, max_sets, SET_SIZE), dtype=torch.int32, device=device
        )
        padded_masks[:, :, 0] = 0
        padded_gather = torch.zeros(
            (2, max_pillars), dtype=torch.int32, device=device
        )
        padded_indices[:, :set_count] = indices.to(torch.int32)
        padded_masks[:, :set_count] = masks.to(torch.int32)
        padded_gather[:, :example_pillars] = gather.to(torch.int32)
        output_indices.append(padded_indices)
        output_masks.append(padded_masks)
        output_gathers.append(padded_gather)

    padded_positions = positions.new_zeros((4, 2, max_pillars, 128))
    padded_positions[:, :, :example_pillars] = positions
    return (
        padded_src,
        output_indices[0],
        output_indices[1],
        output_masks[0],
        output_masks[1],
        output_gathers[0],
        output_gathers[1],
        padded_positions,
    )


def folded_bn(norm):
    scale = norm.weight.detach().cpu().numpy() / np.sqrt(
        norm.running_var.detach().cpu().numpy() + norm.eps
    )
    shift = (
        norm.bias.detach().cpu().numpy()
        - norm.running_mean.detach().cpu().numpy() * scale
    )
    return (
        np.ascontiguousarray(scale.astype(np.float32)),
        np.ascontiguousarray(shift.astype(np.float32)),
    )


def frontend_weights(encoder):
    weights = {}
    for index, layer in enumerate(encoder.vfe.layers):
        scale, shift = folded_bn(layer.norm)
        weights[f"pfn{index}.weight"] = np.ascontiguousarray(
            layer.linear.weight.detach().cpu().numpy().T.astype(np.float32)
        )
        weights[f"pfn{index}.scale"] = scale
        weights[f"pfn{index}.shift"] = shift

    for block in range(4):
        for shift_id in range(2):
            module = encoder.backbone.input_layer.position_layers[block][shift_id]
            first = module.layers[0]
            scale, bn_shift = folded_bn(module.layers[1])
            first_bias = first.bias.detach().cpu().numpy().astype(np.float32)
            prefix = f"position.{block}.{shift_id}"
            weights[prefix + ".first_weight"] = np.ascontiguousarray(
                first.weight.detach().cpu().numpy().T.astype(np.float32)
            )
            weights[prefix + ".first_scale"] = scale
            weights[prefix + ".first_shift"] = np.ascontiguousarray(
                (first_bias * scale + bn_shift).astype(np.float32)
            )
            second = module.layers[3]
            weights[prefix + ".second_weight"] = np.ascontiguousarray(
                second.weight.detach().cpu().numpy().T.astype(np.float32)
            )
            weights[prefix + ".second_bias"] = np.ascontiguousarray(
                second.bias.detach().cpu().numpy().astype(np.float32)
            )
    return weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument("--max-points", type=int, required=True)
    parser.add_argument("--max-pillars", type=int, required=True)
    parser.add_argument("--max-sets", type=int, required=True)
    parser.add_argument("--example-pillars", type=int, default=5200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opset", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this deployment export")
    export_opset = prepare_onnx_export(args.opset)

    model, _, provenance = load_deployment_model(
        args.config,
        args.checkpoint,
        args.allow_random_init,
        args.seed,
    )
    encoder = model.encoders["lidar"]["backbone"]
    validate_encoder_contract(
        encoder, args.max_points, args.max_pillars, args.max_sets
    )
    device = torch.device(args.device)
    backbone = encoder.backbone.to(device).eval()
    neck = encoder.neck.to(device).eval()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    backbone_path = output_dir / "dsvt_padded_backbone.onnx"
    neck_path = output_dir / "dsvt_bev_neck.onnx"
    weights_path = output_dir / "lidar_frontend_weights.npz"
    manifest_path = output_dir / "lidar_trt.manifest.json"

    padded = padded_inputs(
        backbone,
        args.example_pillars,
        args.max_pillars,
        args.max_sets,
        device,
    )
    backbone_wrapper = PaddedDSVTBackbone(backbone).to(device).eval()
    with torch.no_grad():
        transformed = backbone_wrapper(*padded)
    expected_shape = (args.max_pillars, 128)
    if tuple(transformed.shape) != expected_shape:
        raise RuntimeError(
            f"invalid padded DSVT output: expected={expected_shape}, "
            f"actual={tuple(transformed.shape)}"
        )
    with torch.no_grad():
        torch.onnx.export(
            backbone_wrapper,
            padded,
            str(backbone_path),
            input_names=INPUT_NAMES,
            output_names=("transformed_pillars",),
            opset_version=export_opset,
            do_constant_folding=True,
        )

    neck_wrapper = DSVTBEVNeck(neck).to(device).eval()
    dense_bev = torch.randn(1, 128, 360, 360, device=device)
    with torch.no_grad():
        lidar_bev = neck_wrapper(dense_bev)
        torch.onnx.export(
            neck_wrapper,
            (dense_bev,),
            str(neck_path),
            input_names=("dense_bev",),
            output_names=("lidar_bev",),
            opset_version=export_opset,
            do_constant_folding=True,
        )
    if tuple(lidar_bev.shape) != LIDAR_BEV_SHAPE:
        raise RuntimeError(f"invalid LiDAR BEV shape {tuple(lidar_bev.shape)}")

    backbone_graph = load_and_check_onnx(backbone_path, args.opset)
    neck_graph = load_and_check_onnx(neck_path, args.opset)
    np.savez(weights_path, **frontend_weights(encoder))
    manifest = {
        "schema_version": 1,
        "engine": "lidar_raw",
        "model": provenance,
        "capacity": {
            "max_points": args.max_points,
            "max_pillars": args.max_pillars,
            "max_sets_per_shift": args.max_sets,
            "coordinate_capacity": MAX_GRID_PILLARS,
            "set_size": SET_SIZE,
        },
        "inputs": {
            "points": {
                "dtype": "float32",
                "shape": ["N", 5],
                "range": [1, args.max_points],
            }
        },
        "outputs": {
            "lidar_bev": {"dtype": "float32", "shape": list(LIDAR_BEV_SHAPE)},
            "lidar_status": {
                "dtype": "int32",
                "shape": [1],
                "semantics": "0=valid, nonzero=capacity overflow",
            },
        },
        "artifacts": {
            "backbone": {
                "file": backbone_path.name,
                "sha256": sha256_file(backbone_path),
                "nodes": len(backbone_graph.graph.node),
            },
            "neck": {
                "file": neck_path.name,
                "sha256": sha256_file(neck_path),
                "nodes": len(neck_graph.graph.node),
            },
            "frontend_weights": {
                "file": weights_path.name,
                "sha256": sha256_file(weights_path),
            },
        },
        "plugin_abi": {
            "version": 1,
            "creators": [
                "DynamicPillarDecorate:1",
                "DynamicScatterMax:1",
                "TensorBarrier:1",
                "DSVTRotatedSet:1",
                "DSVTDenseScatter:1",
            ],
        },
        "opset": args.opset,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"PASS lidar_trt_artifacts={output_dir} "
        f"max_points={args.max_points} max_pillars={args.max_pillars} "
        f"max_sets={args.max_sets} manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
