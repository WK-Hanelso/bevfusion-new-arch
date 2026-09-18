#!/usr/bin/env python3
"""Check image imports/configs without data; --cuda adds a small GPU ops test."""

import argparse
import importlib
from importlib.metadata import version
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STACKS = {
    "cu113": {
        "python": (3, 8),
        "torch_cuda": "11.3",
        "cuda_arch": None,
        "packages": {
            "torch": "==1.10.1+cu113", "torchvision": "==0.11.2+cu113",
            "mmcv-full": "==1.4.0", "mmdet": "==2.20.0",
            "torch-scatter": "==2.0.9", "numpy": "==1.22.4",
            "numba": "==0.48.0", "torchpack": "==0.3.1",
            "nuscenes-devkit": "==1.1.11", "onnx": "==1.14.1",
            "opencv-python": "==4.8.1.78", "yapf": "==0.32.0",
            "protobuf": "==3.20.3",
        },
    },
    "cu128": {
        "python": (3, 10),
        "torch_cuda": "12.8",
        "cuda_arch": "sm_100",
        "packages": {
            "torch": "==2.7.1+cu128", "torchvision": "==0.22.1+cu128",
            "mmcv-full": "==1.7.2", "mmdet": "==2.28.2",
            "torch-scatter": "==2.1.2", "numpy": "==1.26.4",
            "numba": "==0.61.2", "llvmlite": "==0.44.0",
            "torchpack": "==0.3.1", "nuscenes-devkit": "==1.1.11",
            "onnx": ">=1.14,<2", "opencv-python": ">=4.8,<5",
            "yapf": "==0.32.0", "protobuf": ">=3.20.3,<6",
        },
    },
}
CONFIGS = {
    "camera+lidar/resnet50/convfuser.yaml": ("BEVFusion", 3),
    "camera+lidar/resnet50/convfuser_6cam.yaml": ("BEVFusion", 6),
    "lidar/dsvt_dgf_dal_widthformer_0p3.yaml": ("DynamicBEVFusion", 6),
}
EXTENSIONS = (
    "mmdet3d.ops.spconv.sparse_conv_ext",
    "mmdet3d.ops.bev_pool.bev_pool_ext",
    "mmdet3d.ops.iou3d.iou3d_cuda",
    "mmdet3d.ops.voxel.voxel_layer",
    "mmdet3d.ops.roiaware_pool3d.roiaware_pool3d_ext",
    "mmdet3d.ops.ball_query.ball_query_ext",
    "mmdet3d.ops.knn.knn_ext",
    "mmdet3d.ops.paconv.assign_score_withk_ext",
    "mmdet3d.ops.group_points.group_points_ext",
    "mmdet3d.ops.interpolate.interpolate_ext",
    "mmdet3d.ops.furthest_point_sample.furthest_point_sample_ext",
    "mmdet3d.ops.gather_points.gather_points_ext",
)


def infer_stack():
    torch_version = version("torch")
    if torch_version == "1.10.1+cu113":
        return "cu113"
    if torch_version == "2.7.1+cu128":
        return "cu128"
    # Non-strict development environments still benefit from the import/config
    # checks even when they do not exactly match either Docker image.
    return "cu128" if sys.version_info[:2] == (3, 10) else "cu113"


def version_matches(actual, expected):
    if expected.startswith("==") and "," not in expected:
        return actual == expected[2:]
    # packaging is an explicit cu128 dependency. The legacy cu113 profile only
    # uses exact versions, so it does not gain a new runtime requirement.
    from packaging.specifiers import SpecifierSet
    return actual in SpecifierSet(expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="Require Docker's pinned versions")
    parser.add_argument("--cuda", action="store_true", help="Require GPU and exercise CUDA ops")
    parser.add_argument(
        "--stack", choices=("auto", *STACKS), default="auto",
        help="Version contract to check (default: infer from installed torch)",
    )
    args = parser.parse_args()
    stack_name = infer_stack() if args.stack == "auto" else args.stack
    stack = STACKS[stack_name]
    print(f"stack={stack_name}")
    if args.strict and (
        sys.version_info[:2] != stack["python"] or platform.machine() != "x86_64"
    ):
        python_version = ".".join(str(part) for part in stack["python"])
        raise RuntimeError(
            f"The {stack_name} image requires CPython {python_version} on x86_64"
        )
    for package, expected in stack["packages"].items():
        actual = version(package)
        print(f"{package}={actual}")
        if args.strict and not version_matches(actual, expected):
            raise RuntimeError(f"{package}: expected {expected}, got {actual}")

    import torch
    import torch_scatter
    import mmcv.ops
    import mmdet3d.models
    from mmcv import Config
    from torchpack.utils.config import configs
    from mmdet3d.models.builder import FUSIONMODELS, BACKBONES, FUSERS, HEADS, VTRANSFORMS
    from mmdet3d.utils import recursive_eval

    if args.strict and torch.version.cuda != stack["torch_cuda"]:
        raise RuntimeError(
            f"Expected torch CUDA {stack['torch_cuda']}, got {torch.version.cuda}"
        )
    if args.strict and stack["cuda_arch"]:
        compiled_archs = torch.cuda.get_arch_list()
        print("torch CUDA archs=" + ",".join(compiled_archs))
        if stack["cuda_arch"] not in compiled_archs:
            raise RuntimeError(
                f"torch wheel lacks {stack['cuda_arch']}: {compiled_archs}"
            )
    for module in EXTENSIONS:
        importlib.import_module(module)
    print(f"PASS imports: MMCV ops, torch_scatter, {len(EXTENSIONS)} repository extensions")
    for registry, name in (
        (FUSIONMODELS, "BEVFusion"), (FUSIONMODELS, "DynamicBEVFusion"),
        (BACKBONES, "DSVTLidarEncoder"), (FUSERS, "DepthGFusion"),
        (HEADS, "DALDecoupledHead"), (VTRANSFORMS, "WidthFormerTransform"),
    ):
        if registry.get(name) is None:
            raise RuntimeError("Unregistered module: " + name)
    previous_cwd = Path.cwd()
    try:
        os.chdir(ROOT)
        for relative, (model_type, camera_count) in CONFIGS.items():
            configs.clear()
            path = "configs/nuscenes/det/transfusion/secfpn/" + relative
            configs.load(path, recursive=True)
            cfg = Config(recursive_eval(configs), filename=path)
            if cfg.model.type != model_type or len(cfg.camera_names) != camera_count:
                raise RuntimeError("Unexpected model/camera contract: " + path)
            print(f"PASS config: {model_type}, {camera_count} cameras, {path}")
    finally:
        configs.clear()
        os.chdir(previous_cwd)
    if args.cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable: check docker --gpus and host NVIDIA runtime")
        src = torch.tensor([1.0, 3.0, 2.0], device="cuda")
        idx = torch.tensor([0, 0, 1], device="cuda")
        result = torch_scatter.scatter_max(src, idx, dim=0)[0]
        torch.testing.assert_close(result.cpu(), torch.tensor([3.0, 2.0]))
        from mmdet3d.ops.voxel import Voxelization
        voxelize = Voxelization(voxel_size=[1, 1, 1],
                               point_cloud_range=[0, 0, 0, 4, 4, 4],
                               max_num_points=5, max_voxels=(10, 10))
        points = torch.tensor([[0.1, 0.1, 0.1, 1, 0],
                               [0.2, 0.1, 0.1, 1, 0]], device="cuda")
        _, _, counts = voxelize(points)
        if counts.sum().item() != 2:
            raise RuntimeError("CUDA voxelization lost valid points")
        torch.cuda.synchronize()
        print(f"PASS CUDA scatter/voxelization: {torch.cuda.get_device_name()}")
    print("PASS environment check (no dataset training or accuracy evaluation)")


if __name__ == "__main__":
    main()
