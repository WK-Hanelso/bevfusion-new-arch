#!/usr/bin/env python
"""Diagnose *where* predictions land relative to ground truth.

Runs a checkpoint on the first N validation samples (single process, one GPU)
and reports, for boxes above a score threshold, the distance to the nearest
ground-truth center under several candidate coordinate transforms:

    identity, flip_x, flip_y, swap_xy, scale_x2, scale_x0.5, rot180

If one transform clearly beats identity, the training targets and the decoded
boxes disagree on a coordinate convention. If identity wins but the distances
are still large (>1 m), the model is simply not trained yet / diverged.

Usage (server, conda env bevfusion-b200, from repo root):
    CUDA_VISIBLE_DEVICES=0 python tools/ablation/diag_pred_offset.py \
        configs/nuscenes/det/ablation/dsvt1_wf1_gf0_dal0.yaml \
        experiments/runs/screening/C1/checkpoints/C1/epoch_1.pth --num 40
"""

import argparse
import copy

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate, scatter
from mmcv.runner import load_checkpoint
from torchpack.utils.config import configs

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--num", type=int, default=40)
    parser.add_argument("--score", type=float, default=0.1)
    parser.add_argument("--dataset-root", default=None)
    return parser.parse_args()


TRANSFORMS = {
    "identity": lambda c: c,
    "flip_x": lambda c: c * np.array([-1.0, 1.0]),
    "flip_y": lambda c: c * np.array([1.0, -1.0]),
    "swap_xy": lambda c: c[:, ::-1],
    "rot180": lambda c: -c,
    "scale_x2": lambda c: c * 2.0,
    "scale_x0.5": lambda c: c * 0.5,
}


def main():
    args = parse_args()
    configs.load(args.config, recursive=True)
    if args.dataset_root:
        configs.update(["--dataset_root", args.dataset_root])
    cfg = Config(recursive_eval(configs), filename=args.config)

    cfg.model.pretrained = None
    test_cfg = copy.deepcopy(cfg.data.val)
    test_cfg.test_mode = True
    test_cfg.pop("samples_per_gpu", None)
    dataset = build_dataset(test_cfg)

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = MMDataParallel(model.cuda(), device_ids=[0]).eval()

    per_transform = {name: [] for name in TRANSFORMS}
    n_pred_total = 0
    n_gt_total = 0
    score_stats = []
    size_ratio = []

    with torch.no_grad():
        for index in range(min(args.num, len(dataset))):
            data = collate([dataset[index]], samples_per_gpu=1)
            data = scatter(data, [0])[0]
            result = model(return_loss=False, rescale=True, **data)[0]
            boxes = result["boxes_3d"].tensor.cpu().numpy()
            scores = result["scores_3d"].cpu().numpy()
            score_stats.append(scores)
            keep = scores >= args.score
            boxes = boxes[keep]
            gt = dataset.get_ann_info(index)["gt_bboxes_3d"].tensor.cpu().numpy()
            n_pred_total += len(boxes)
            n_gt_total += len(gt)
            if len(boxes) == 0 or len(gt) == 0:
                continue
            gt_xy = gt[:, :2]
            pred_xy = boxes[:, :2]
            for name, fn in TRANSFORMS.items():
                moved = np.ascontiguousarray(fn(pred_xy))
                d = np.linalg.norm(moved[:, None, :] - gt_xy[None, :, :], axis=-1)
                per_transform[name].append(d.min(axis=1))
            # size sanity: pred (l,w) vs nearest gt (l,w)
            d = np.linalg.norm(pred_xy[:, None, :] - gt_xy[None, :, :], axis=-1)
            nearest = d.argmin(axis=1)
            size_ratio.append(boxes[:, 3:5] / np.maximum(gt[nearest, 3:5], 1e-3))

    all_scores = np.concatenate(score_stats) if score_stats else np.zeros(0)
    print(f"samples={min(args.num, len(dataset))} gt_boxes={n_gt_total} "
          f"pred_boxes(score>={args.score})={n_pred_total}")
    if all_scores.size:
        q = np.quantile(all_scores, [0.5, 0.9, 0.99, 1.0])
        print(f"score quantiles p50/p90/p99/max = {q[0]:.3f}/{q[1]:.3f}/{q[2]:.3f}/{q[3]:.3f}")
    print("transform     | mean_dist(m) | median | <=1m rate | <=2m rate")
    for name, chunks in per_transform.items():
        if not chunks:
            print(f"{name:13s} | no predictions")
            continue
        d = np.concatenate(chunks)
        print(f"{name:13s} | {d.mean():11.3f} | {np.median(d):6.3f} | "
              f"{(d <= 1.0).mean():9.3f} | {(d <= 2.0).mean():9.3f}")
    if size_ratio:
        r = np.concatenate(size_ratio)
        print(f"pred/gt size ratio (l,w) median = {np.median(r[:, 0]):.3f}, {np.median(r[:, 1]):.3f}")


if __name__ == "__main__":
    main()
