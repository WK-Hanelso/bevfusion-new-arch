#!/usr/bin/env python3
"""Export DSVT blocks, dense pillar scatter, and the BEV neck to ONNX."""

import argparse
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import onnx
import torch
from torch import nn

from deployment.onnx.export_device import require_export_device
from deployment.onnx.model_contract import (
    DEFAULT_CONFIG,
    LIDAR_BEV_SHAPE,
    load_deployment_model,
    write_manifest,
)
from deployment.onnx.onnx_compat import (
    load_and_check_onnx,
    prepare_onnx_export,
)


def load_backbone_exporter():
    path = Path(__file__).resolve().parent / "dsvt_backbone.py"
    spec = importlib.util.spec_from_file_location("dsvt_backbone_export", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXPORT = load_backbone_exporter()


class DSVTBEVDeployWrapper(nn.Module):
    def __init__(self, backbone, scatter, neck):
        super().__init__()
        self.transformer = EXPORT.DSVTDeployWrapper(backbone)
        self.scatter = scatter
        self.neck = neck

    def forward(
        self,
        src,
        set_indices_shift_0,
        set_indices_shift_1,
        set_masks_shift_0,
        set_masks_shift_1,
        gather_shift_0,
        gather_shift_1,
        position_embeddings,
        coords,
    ):
        transformed = self.transformer(
            src,
            set_indices_shift_0,
            set_indices_shift_1,
            set_masks_shift_0,
            set_masks_shift_1,
            gather_shift_0,
            gather_shift_1,
            position_embeddings,
        )
        return self.neck(self.scatter(transformed, coords, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--pillars", type=int, default=5200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opset", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-cpu-only", action="store_true")
    args = parser.parse_args()
    export_provenance = require_export_device(args.device, args.allow_cpu_only)
    export_opset = prepare_onnx_export(args.opset)

    complete_model, _, provenance = load_deployment_model(
        args.config, args.checkpoint, args.allow_random_init, args.seed
    )
    provenance.update(export_provenance)
    lidar_encoder = complete_model.encoders["lidar"]["backbone"]
    device = torch.device(args.device)
    backbone = lidar_encoder.backbone.to(device).eval()
    scatter = lidar_encoder.scatter.to(device).eval()
    neck = lidar_encoder.neck.to(device).eval()
    wrapper = DSVTBEVDeployWrapper(backbone, scatter, neck).to(device).eval()
    backbone_inputs, coords = EXPORT.make_inputs(
        backbone, args.pillars, device
    )
    inputs = (*backbone_inputs, coords)
    with torch.no_grad():
        transformed = backbone(backbone_inputs[0], coords)
        expected = neck(scatter(transformed, coords, 1))
        actual = wrapper(*inputs)
    maximum_error = (expected - actual).abs().max().item()
    if maximum_error != 0.0:
        raise RuntimeError(f"deployment wrapper mismatch: max_abs={maximum_error}")

    input_names = (
        "src",
        "set_indices_shift_0",
        "set_indices_shift_1",
        "set_masks_shift_0",
        "set_masks_shift_1",
        "gather_shift_0",
        "gather_shift_1",
        "position_embeddings",
        "coords",
    )
    if args.reference_dir is not None:
        EXPORT.save_reference(args.reference_dir, input_names, inputs, actual)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = {
        "src": {0: "pillar_count"},
        "set_indices_shift_0": {1: "set_count_shift_0"},
        "set_indices_shift_1": {1: "set_count_shift_1"},
        "set_masks_shift_0": {1: "set_count_shift_0"},
        "set_masks_shift_1": {1: "set_count_shift_1"},
        "gather_shift_0": {1: "pillar_count"},
        "gather_shift_1": {1: "pillar_count"},
        "position_embeddings": {2: "pillar_count"},
        "coords": {0: "pillar_count"},
    }
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            inputs,
            str(args.output),
            input_names=input_names,
            output_names=("lidar_bev",),
            dynamic_axes=dynamic_axes,
            opset_version=export_opset,
            do_constant_folding=True,
        )
    graph = load_and_check_onnx(args.output, args.opset)
    operation_types = sorted({node.op_type for node in graph.graph.node})
    manifest = args.manifest or args.output.with_suffix(".manifest.json")
    write_manifest(
        manifest,
        "dsvt_bev",
        provenance,
        inputs={name: {
            "dtype": str(value.dtype).replace("torch.", ""),
            "shape": list(value.shape),
        } for name, value in zip(input_names, inputs)},
        outputs={"lidar_bev": {
            "dtype": "float32", "shape": list(LIDAR_BEV_SHAPE),
        }},
        extra={
            "onnx": str(args.output.resolve()),
            "opset": args.opset,
            "production_engine_b": False,
            "onnx_input_boundary": "pillar features + DSVT topology + coordinates",
            "excluded_prefix": "raw points -> dynamic pillar VFE -> topology",
        },
    )
    print(
        f"PASS onnx={args.output} size={args.output.stat().st_size} "
        f"nodes={len(graph.graph.node)} max_abs={maximum_error} "
        f"output={tuple(actual.shape)} ops={operation_types} manifest={manifest}"
    )


if __name__ == "__main__":
    main()
