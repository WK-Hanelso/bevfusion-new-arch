#!/usr/bin/env python3
"""Export ResNet-34 + LSSFPN + WidthFormer as the camera TensorRT graph."""

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

from deployment.onnx.model_contract import (
    CAMERA_BEV_SHAPE,
    DEFAULT_CONFIG,
    load_deployment_model,
    write_manifest,
)
from deployment.onnx.onnx_compat import (
    load_and_check_onnx,
    prepare_onnx_export,
)


class CameraBEVDeployWrapper(nn.Module):
    def __init__(self, camera_encoder):
        super().__init__()
        self.backbone = camera_encoder["backbone"]
        self.neck = camera_encoder["neck"]
        self.transform = camera_encoder["vtransform"]

    def forward(self, images, geometry):
        batch, cameras = images.shape[:2]
        features = self.neck(self.backbone(images.flatten(0, 1)))
        features = features.reshape(batch, cameras, *features.shape[1:])
        return self.transform.forward_with_geometry(features, geometry)


def make_geometry(device):
    depth = torch.arange(1.0, 60.0, device=device).view(1, 1, 59, 1, 1)
    x = torch.linspace(-40.0, 40.0, 44, device=device).view(1, 1, 1, 1, 44)
    y = torch.linspace(-15.0, 15.0, 16, device=device).view(1, 1, 1, 16, 1)
    angle = torch.linspace(-1.2, 1.2, 6, device=device).view(1, 6, 1, 1, 1)
    xx = depth * torch.cos(angle) + x
    yy = depth * torch.sin(angle) + y
    zz = torch.zeros_like(xx + yy)
    return torch.stack((
        xx.expand(1, 6, 59, 16, 44),
        yy.expand(1, 6, 59, 16, 44),
        zz.expand(1, 6, 59, 16, 44),
    ), dim=-1).contiguous()


def save_reference(directory, values):
    directory.mkdir(parents=True, exist_ok=True)
    for name, tensor in values.items():
        array = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
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
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this deployment export")
    export_opset = prepare_onnx_export(args.opset)

    complete_model, _, provenance = load_deployment_model(
        args.config, args.checkpoint, args.allow_random_init, args.seed
    )
    device = torch.device(args.device)
    model = CameraBEVDeployWrapper(
        complete_model.encoders["camera"]
    ).to(device).eval()
    images = torch.randn(1, 6, 3, 256, 704, device=device)
    geometry = make_geometry(images.device)
    with torch.no_grad():
        camera_bev = model(images, geometry)
    if tuple(camera_bev.shape) != CAMERA_BEV_SHAPE:
        raise RuntimeError(f"invalid camera BEV shape {tuple(camera_bev.shape)}")
    if args.reference_dir:
        save_reference(args.reference_dir, {
            "images": images, "geometry": geometry,
            "camera_bev_pytorch": camera_bev,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model, (images, geometry), str(args.output),
            input_names=("images", "geometry"), output_names=("camera_bev",),
            opset_version=export_opset, do_constant_folding=True,
        )
    graph = load_and_check_onnx(args.output, args.opset)
    manifest = args.manifest or args.output.with_suffix(".manifest.json")
    write_manifest(
        manifest,
        "camera_bev",
        provenance,
        inputs={
            "images": {"dtype": "float32", "shape": list(images.shape)},
            "geometry": {"dtype": "float32", "shape": list(geometry.shape)},
        },
        outputs={
            "camera_bev": {
                "dtype": "float32",
                "shape": list(CAMERA_BEV_SHAPE),
            }
        },
        extra={"onnx": str(args.output.resolve()), "opset": args.opset},
    )
    print(
        f"PASS onnx={args.output} size={args.output.stat().st_size} "
        f"nodes={len(graph.graph.node)} output={tuple(camera_bev.shape)} "
        f"manifest={manifest}"
    )


if __name__ == "__main__":
    main()
