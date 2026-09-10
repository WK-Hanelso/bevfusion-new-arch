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

EXPECTED = {
    "torch": "1.10.1+cu113", "torchvision": "0.11.2+cu113",
    "mmcv-full": "1.4.0", "mmdet": "2.20.0", "torch-scatter": "2.0.9",
    "numpy": "1.22.4", "numba": "0.48.0", "torchpack": "0.3.1",
    "nuscenes-devkit": "1.1.11", "onnx": "1.14.1",
    "opencv-python": "4.8.1.78", "yapf": "0.32.0", "protobuf": "3.20.3",
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="Require Docker's pinned versions")
    parser.add_argument("--cuda", action="store_true", help="Require GPU and exercise CUDA ops")
    args = parser.parse_args()
    if args.strict and (sys.version_info[:2] != (3, 8) or platform.machine() != "x86_64"):
        raise RuntimeError("This training image requires CPython 3.8 on x86_64")
    for package, expected in EXPECTED.items():
        actual = version(package)
        print(f"{package}={actual}")
        if args.strict and actual != expected:
            raise RuntimeError(f"{package}: expected {expected}, got {actual}")

    import torch
    import torch_scatter
    import mmcv.ops
    import mmdet3d.models
    from mmcv import Config
    from torchpack.utils.config import configs
    from mmdet3d.models.builder import FUSIONMODELS, BACKBONES, FUSERS, HEADS, VTRANSFORMS
    from mmdet3d.utils import recursive_eval

    if args.strict and torch.version.cuda != "11.3":
        raise RuntimeError(f"Expected torch CUDA 11.3, got {torch.version.cuda}")
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
