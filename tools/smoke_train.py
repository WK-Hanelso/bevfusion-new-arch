#!/usr/bin/env python3
"""Run one real PyTorch training step from a standard BEVFusion config."""

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import build_optimizer, wrap_fp16_model
from torchpack.utils.config import configs


BEVFUSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BEVFUSION_ROOT))

from mmdet3d.datasets import build_dataloader, build_dataset  # noqa: E402
from mmdet3d.models import build_model  # noqa: E402
from mmdet3d.utils import recursive_eval  # noqa: E402


def _rewrite_dataset_root(value, old_root, new_root):
    """Rewrite resolved dataset paths without changing the model config."""
    if isinstance(value, dict):
        for key in list(value):
            value[key] = _rewrite_dataset_root(value[key], old_root, new_root)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = _rewrite_dataset_root(item, old_root, new_root)
    elif isinstance(value, tuple):
        value = tuple(
            _rewrite_dataset_root(item, old_root, new_root) for item in value
        )
    elif isinstance(value, str) and value.startswith(old_root):
        value = new_root + value[len(old_root):]
    return value


def _load_training_config(config_path, dataroot):
    relative = config_path.relative_to(BEVFUSION_ROOT)
    original_cwd = Path.cwd()
    try:
        os.chdir(str(BEVFUSION_ROOT))
        configs.clear()
        configs.load(str(relative), recursive=True)
        cfg = Config(recursive_eval(configs), filename=str(relative))
    finally:
        os.chdir(str(original_cwd))

    old_root = str(cfg.dataset_root)
    new_root = str(dataroot.resolve()) + "/"
    _rewrite_dataset_root(cfg._cfg_dict, old_root, new_root)
    cfg.data.samples_per_gpu = 1
    cfg.data.workers_per_gpu = 0
    return cfg


def _remove_map_pipeline(cfg):
    """Remove map-label loading for object-only mini smoke tests."""
    pipeline = [
        stage
        for stage in cfg.train_pipeline
        if stage.get("type") != "LoadBEVSegmentation"
    ]
    for stage in pipeline:
        if stage.get("type") == "Collect3D":
            stage["keys"] = [
                key for key in stage["keys"] if key != "gt_masks_bev"
            ]
    cfg.train_pipeline = pipeline
    cfg.data.train.dataset.pipeline = pipeline


def _gradient_count(module):
    return sum(
        parameter.grad is not None
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--dataroot", type=Path, default=Path("data/nuscenes"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--forward-only",
        action="store_true",
        help="compute losses but skip backward and optimizer.step",
    )
    parser.add_argument(
        "--object-only",
        action="store_true",
        help="skip LoadBEVSegmentation when mini map-expansion files are absent",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="apply MMCV mixed-precision wrapping used by the normal trainer",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("the PyTorch variant smoke test requires CUDA")
    train_info = args.dataroot.expanduser().resolve() / "nuscenes_infos_train.pkl"
    if not train_info.is_file():
        raise FileNotFoundError(train_info)

    config_path = args.config.expanduser()
    if not config_path.is_absolute():
        config_path = BEVFUSION_ROOT / config_path
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    cfg = _load_training_config(config_path, args.dataroot.expanduser())
    if args.object_only:
        _remove_map_pipeline(cfg)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(args.device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(args.device)

    timings = {}
    started = time.perf_counter()
    dataset = build_dataset(cfg.data.train)
    timings["dataset"] = time.perf_counter() - started

    started = time.perf_counter()
    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        num_gpus=1,
        dist=False,
        shuffle=False,
    )
    data = next(iter(loader))
    timings["batch"] = time.perf_counter() - started

    started = time.perf_counter()
    module = build_model(cfg.model).cuda(args.device)
    model = MMDataParallel(module, device_ids=[args.device])
    if args.fp16:
        wrap_fp16_model(model)
    model.train()
    optimizer = build_optimizer(module, cfg.optimizer)
    optimizer.zero_grad()
    torch.cuda.synchronize(args.device)
    timings["model"] = time.perf_counter() - started

    started = time.perf_counter()
    outputs = model(return_loss=True, **data)
    losses = [
        value.mean()
        for key, value in outputs.items()
        if key.startswith("loss/")
    ]
    if not losses:
        raise RuntimeError("model returned no trainable losses")
    loss = sum(losses)
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite total loss: {loss.item()}")
    torch.cuda.synchronize(args.device)
    timings["forward"] = time.perf_counter() - started

    gradients = {}
    if not args.forward_only:
        started = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize(args.device)
        timings["backward"] = time.perf_counter() - started
        gradients = {
            "all": _gradient_count(module),
            "camera": _gradient_count(module.encoders["camera"]),
            "lidar": _gradient_count(module.encoders["lidar"]),
            "fusion": _gradient_count(module.fuser),
            "head": _gradient_count(module.heads["object"]),
        }
        missing = [name for name, count in gradients.items() if count == 0]
        if missing:
            raise RuntimeError(f"no gradients in required modules: {missing}")

        grad_cfg = cfg.optimizer_config.get("grad_clip")
        if grad_cfg:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_cfg.max_norm,
                norm_type=grad_cfg.norm_type,
            )
        started = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize(args.device)
        timings["optimizer"] = time.perf_counter() - started

    points = data["points"].data[0][0]
    images = data["img"].data[0]
    boxes = data["gt_bboxes_3d"].data[0][0]
    loss_values = {
        key: round(float(value.detach().mean()), 6)
        for key, value in outputs.items()
    }
    timing_values = {key: round(value, 4) for key, value in timings.items()}
    print(
        "PASS "
        f"config={config_path.relative_to(BEVFUSION_ROOT)} "
        f"model={type(module).__name__} "
        f"views={images.shape[1]} points={points.shape[0]} boxes={len(boxes)} "
        f"parameters={sum(p.numel() for p in module.parameters())} "
        f"total_loss={float(loss.detach()):.6f} gradients={gradients} "
        f"peak_cuda_gib={torch.cuda.max_memory_allocated(args.device) / 1024**3:.3f} "
        f"timings_s={timing_values} losses={loss_values}"
    )


if __name__ == "__main__":
    main()
