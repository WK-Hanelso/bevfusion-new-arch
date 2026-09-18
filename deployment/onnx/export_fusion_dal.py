#!/usr/bin/env python3
"""Export Depth-GFusion + SECOND/FPN + DAL as the final TensorRT graph."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import onnx
import torch
from torch import nn
from torch.onnx.utils import ONNXCheckerError

from deployment.onnx.export_device import require_export_device
from deployment.onnx.model_contract import (
    CAMERA_BEV_SHAPE,
    DEFAULT_CONFIG,
    LIDAR_BEV_SHAPE,
    load_deployment_model,
    write_manifest,
)
from deployment.onnx.onnx_compat import (
    load_and_check_onnx,
    prepare_onnx_export,
)


OUTPUT_NAMES = ("boxes", "scores", "labels")


def decode_dal_outputs(result):
    query_labels = result["query_labels"].long()
    one_hot = torch.nn.functional.one_hot(
        query_labels, num_classes=10
    ).permute(0, 2, 1)
    scores = (
        result["heatmap"].sigmoid()
        * result["query_heatmap_score"]
        * one_hot
    )
    final_scores, final_labels = scores.max(dim=1)
    center = result["center"]
    metric_x = center[:, 0:1] * 0.6 - 54.0
    metric_y = center[:, 1:2] * 0.6 - 54.0
    dimensions = result["dim"].exp()
    bottom_height = result["height"] - dimensions[:, 2:3] * 0.5
    rotation = torch.atan2(result["rot"][:, 0:1], result["rot"][:, 1:2])
    boxes = torch.cat(
        (metric_x, metric_y, bottom_height, dimensions, rotation, result["vel"]),
        dim=1,
    ).permute(0, 2, 1)
    return boxes, final_scores, final_labels.to(torch.int32)


class FusionDALDeployWrapper(nn.Module):
    def __init__(self, complete_model):
        super().__init__()
        self.fuser = complete_model.fuser
        self.backbone = complete_model.decoder["backbone"]
        self.neck = complete_model.decoder["neck"]
        self.head = complete_model.heads["object"]

    def forward(self, camera_bev, lidar_bev):
        fused = self.fuser(camera_bev, lidar_bev)
        decoded = self.neck(self.backbone(fused))[0]
        result = self.head(decoded, lidar_bev)[0][0]
        return decode_dal_outputs(result)


def save_reference(directory, values):
    directory.mkdir(parents=True, exist_ok=True)
    for name, tensor in values.items():
        array = tensor.detach().cpu().numpy()
        if array.dtype.kind == "f":
            array = array.astype(np.float32, copy=False)
        elif array.dtype.kind in "iu":
            array = array.astype(np.int32, copy=False)
        array.tofile(directory / f"{name}.bin")
        np.save(directory / f"{name}.npy", array)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
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
    device = torch.device(args.device)
    model = FusionDALDeployWrapper(complete_model).to(device).eval()
    camera_bev = torch.randn(*CAMERA_BEV_SHAPE, device=device)
    lidar_bev = torch.randn(*LIDAR_BEV_SHAPE, device=device)
    with torch.no_grad():
        outputs = model(camera_bev, lidar_bev)
    if not all(torch.isfinite(x).all() for x in outputs if x.is_floating_point()):
        raise RuntimeError("non-finite fusion/DAL reference output")
    if args.reference_dir:
        values = {"camera_bev": camera_bev, "lidar_bev": lidar_bev}
        values.update(dict(zip(OUTPUT_NAMES, outputs)))
        save_reference(args.reference_dir, values)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with torch.no_grad():
            torch.onnx.export(
                model, (camera_bev, lidar_bev), str(args.output),
                input_names=("camera_bev", "lidar_bev"),
                output_names=OUTPUT_NAMES,
                opset_version=export_opset,
                do_constant_folding=True,
            )
    except ONNXCheckerError as error:
        expected_legacy_error = (
            export_opset < args.opset
            and "GridSample" in str(error)
            and f"domain_version of {export_opset}" in str(error)
            and args.output.is_file()
        )
        if not expected_legacy_error:
            raise
    graph = load_and_check_onnx(args.output, args.opset)
    manifest = args.manifest or args.output.with_suffix(".manifest.json")
    output_specs = {
        name: {"dtype": str(value.dtype).replace("torch.", ""),
               "shape": list(value.shape)}
        for name, value in zip(OUTPUT_NAMES, outputs)
    }
    write_manifest(
        manifest,
        "fusion_dal",
        provenance,
        inputs={
            "camera_bev": {"dtype": "float32", "shape": list(CAMERA_BEV_SHAPE)},
            "lidar_bev": {"dtype": "float32", "shape": list(LIDAR_BEV_SHAPE)},
        },
        outputs=output_specs,
        extra={"onnx": str(args.output.resolve()), "opset": args.opset},
    )
    print(
        f"PASS onnx={args.output} size={args.output.stat().st_size} "
        f"nodes={len(graph.graph.node)} outputs="
        + ",".join(f"{n}:{tuple(v.shape)}" for n, v in zip(OUTPUT_NAMES, outputs))
        + f" manifest={manifest}"
    )


if __name__ == "__main__":
    main()
