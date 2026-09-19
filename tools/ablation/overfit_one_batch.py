#!/usr/bin/env python
"""Single-batch overfit probe: does the training loop actually learn?

Builds the model from a config, takes ONE training sample (repeated to the
requested batch size), and optimises on that same batch for K steps with the
config's optimizer (constant lr, no schedule). A healthy pipeline drives the
object losses down sharply (heatmap loss well below 0.5 within ~100 steps).

Runs on CPU or GPU, no DDP, no runner hooks: this isolates model + loss +
optimizer from the launcher/runner/DDP layers. Compare the printed curve
between stacks (laptop docker torch 1.10 vs B200 torch 2.7).

Usage:
    python tools/ablation/overfit_one_batch.py <config> [--steps 100]
        [--batch 2] [--index 0] [--dataset-root data/nuscenes/] [--load-from X]
"""

import argparse
import time

import torch
from mmcv import Config
from mmcv.parallel import collate
from mmcv.runner import load_checkpoint
from torchpack.utils.config import configs

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import train_torchrun as _shim  # noqa: E402  mmcv 1.x <-> torch>=2.1 compatibility patches

_shim._patch_yapf()
_shim._patch_mmcv_get_stream()
_shim._patch_mmcv_ddp_forward()

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--load-from", default=None)
    parser.add_argument("--lr", type=float, default=None, help="override optimizer lr")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    configs.load(args.config, recursive=True)
    if args.dataset_root:
        configs.update(["--dataset_root", args.dataset_root])
    cfg = Config(recursive_eval(configs), filename=args.config)
    torch.manual_seed(0)

    dataset = build_dataset(cfg.data.train)
    # CBGSDataset wraps the real dataset; take a fixed sample from it.
    sample = dataset[args.index]
    data = collate([sample] * args.batch, samples_per_gpu=args.batch)

    model = build_model(
        cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg")
    )
    if args.load_from:
        load_checkpoint(model, args.load_from, map_location="cpu")
    use_cuda = torch.cuda.is_available() and not args.cpu
    from mmcv.parallel import MMDataParallel

    if use_cuda:
        model = MMDataParallel(model.cuda(), device_ids=[0])
    else:
        from mmcv.parallel import DataContainer

        # CPU: unwrap the per-GPU chunk 0 of every DataContainer by hand.
        data = {
            k: (v.data[0] if isinstance(v, DataContainer) else v)
            for k, v in data.items()
        }
    model.train()

    opt_cfg = dict(cfg.optimizer)
    opt_type = opt_cfg.pop("type")
    if args.lr is not None:
        opt_cfg["lr"] = args.lr
    optimizer = getattr(torch.optim, opt_type)(model.parameters(), **opt_cfg)
    print(f"device={'cuda' if use_cuda else 'cpu'} optimizer={opt_type} {opt_cfg} "
          f"batch={args.batch} steps={args.steps} sample_index={args.index}")

    t0 = time.time()
    for step in range(1, args.steps + 1):
        # Same entry point the mmcv runner uses: model.train_step -> _parse_losses
        outputs = model.train_step(data, None)
        total = outputs["loss"]
        optimizer.zero_grad()
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 35.0)
        optimizer.step()
        if step in (1, 2, 5, 10) or step % 10 == 0:
            log_vars = outputs["log_vars"]
            parts = " ".join(
                f"{k.split('/')[-1]}={v:.4f}" for k, v in log_vars.items() if k != "loss"
            )
            print(
                f"step {step:4d} total={total.item():.4f} {parts} grad_norm={float(grad_norm):.2f} "
                f"t={time.time() - t0:.0f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
