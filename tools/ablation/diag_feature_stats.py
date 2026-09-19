#!/usr/bin/env python
"""Per-stage feature statistics for a trained checkpoint on one validation
sample: is any branch dead (constant / zero / NaN)?

Hooks: lidar encoder output, camera vtransform output, fuser output, decoder
neck output, object-head heatmap logits. Prints mean / std / |max| / zero-frac
/ nan-frac for each, for (a) the checkpoint and (b) the same model with only
the pretrained DSVT weights (or random init) for reference.

Usage:
    CUDA_VISIBLE_DEVICES=0 python tools/ablation/diag_feature_stats.py <config> <ckpt> [--index 0] [--load-from-dsvt X]
"""

import argparse
import copy
import os
import sys

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate, scatter
from mmcv.runner import load_checkpoint
from torchpack.utils.config import configs

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import train_torchrun as _shim  # noqa: E402

_shim._patch_yapf()
_shim._patch_mmcv_get_stream()
_shim._patch_mmcv_ddp_forward()

from mmdet3d.datasets import build_dataset  # noqa: E402
from mmdet3d.models import build_model  # noqa: E402
from mmdet3d.utils import recursive_eval  # noqa: E402


def stats(name, t):
    if isinstance(t, (list, tuple)):
        t = t[0]
    if isinstance(t, dict):
        for k, v in t.items():
            if torch.is_tensor(v):
                stats(f"{name}.{k}", v)
        return
    if not torch.is_tensor(t):
        print(f"{name:34s} <{type(t).__name__}>")
        return
    f = t.detach().float()
    nan = torch.isnan(f).float().mean().item()
    f = torch.nan_to_num(f)
    print(f"{name:34s} shape={tuple(t.shape)} mean={f.mean().item():+.4f} std={f.std().item():.4f} "
          f"|max|={f.abs().max().item():.3f} zero={(f == 0).float().mean().item():.3f} nan={nan:.3f}")


def attach(model, label):
    handles = []
    targets = {
        "lidar.backbone": model.encoders["lidar"]["backbone"] if "lidar" in model.encoders else None,
        "camera.vtransform": model.encoders["camera"]["vtransform"] if "camera" in model.encoders else None,
        "fuser": model.fuser,
        "decoder.neck": model.decoder["neck"],
        "head.heatmap_head": getattr(model.heads["object"], "heatmap_head", None),
    }
    for name, module in targets.items():
        if module is None:
            continue
        handles.append(module.register_forward_hook(
            lambda m, i, o, n=f"[{label}] {name}": stats(n, o)))
    return handles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--load-from-dsvt", default=None)
    parser.add_argument("--train-mode", action="store_true", help="run hooks in train() mode (BN batch stats)")
    args = parser.parse_args()

    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)
    cfg.model.pretrained = None
    test_cfg = copy.deepcopy(cfg.data.val)
    test_cfg.test_mode = True
    test_cfg.pop("samples_per_gpu", None)
    dataset = build_dataset(test_cfg)
    data = scatter(collate([dataset[args.index]], samples_per_gpu=1), [0])[0]

    for label, ckpt in (("trained", args.checkpoint), ("init", args.load_from_dsvt)):
        model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
        if ckpt:
            load_checkpoint(model, ckpt, map_location="cpu")
        model = model.cuda()
        model.train(args.train_mode)
        handles = attach(model, label)
        wrapped = MMDataParallel(model, device_ids=[0])
        with torch.no_grad():
            result = wrapped(return_loss=False, rescale=True, **data)[0]
        scores = result["scores_3d"]
        print(f"[{label}] scores: n={scores.numel()} max={scores.max().item():.4f} "
              f"n>=0.1={(scores >= 0.1).sum().item()}")
        for h in handles:
            h.remove()
        print()


if __name__ == "__main__":
    main()
