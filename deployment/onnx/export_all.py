#!/usr/bin/env python3
"""Export the complete DynamicBEVFusion A/B/C deployment artifacts."""

import argparse
import subprocess
import sys
from pathlib import Path

from deployment.onnx.model_contract import DEFAULT_CONFIG


SCRIPT_DIR = Path(__file__).resolve().parent


def run_export(script, output, args, reference_name):
    command = [
        sys.executable,
        str(SCRIPT_DIR / script),
        "--output",
        str(output),
        "--config",
        str(args.config),
        "--device",
        args.device,
        "--opset",
        str(args.opset),
        "--seed",
        str(args.seed),
    ]
    if args.checkpoint is not None:
        command.extend(("--checkpoint", str(args.checkpoint)))
    if args.allow_random_init:
        command.append("--allow-random-init")
    if args.reference_dir is not None:
        command.extend(
            ("--reference-dir", str(args.reference_dir / reference_name))
        )
    if script == "export_dsvt_bev.py":
        command.extend(("--pillars", str(args.pillars)))
    subprocess.run(command, check=True)


def run_lidar_trt_artifact_export(args):
    command = [
        sys.executable,
        str(SCRIPT_DIR / "export_lidar_trt_artifacts.py"),
        "--output-dir",
        str(args.output_dir / "lidar_trt"),
        "--config",
        str(args.config),
        "--device",
        args.device,
        "--opset",
        str(args.opset),
        "--seed",
        str(args.seed),
        "--example-pillars",
        str(args.pillars),
        "--max-points",
        str(args.max_points),
        "--max-pillars",
        str(args.max_pillars),
        "--max-sets",
        str(args.max_sets),
    ]
    if args.checkpoint is not None:
        command.extend(("--checkpoint", str(args.checkpoint)))
    if args.allow_random_init:
        command.append("--allow-random-init")
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--pillars", type=int, default=5200)
    parser.add_argument("--max-points", type=int, required=True)
    parser.add_argument("--max-pillars", type=int, required=True)
    parser.add_argument("--max-sets", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opset", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.checkpoint is None and not args.allow_random_init:
        parser.error(
            "--checkpoint is required; use --allow-random-init only for "
            "a diagnostic export"
        )
    args.config = args.config.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.reference_dir is not None:
        args.reference_dir.mkdir(parents=True, exist_ok=True)

    run_export(
        "export_camera_bev.py",
        args.output_dir / "camera_bev.onnx",
        args,
        "camera_bev",
    )
    run_lidar_trt_artifact_export(args)
    run_export(
        "export_fusion_dal.py",
        args.output_dir / "fusion_dal.onnx",
        args,
        "fusion_dal",
    )
    print(f"PASS output_dir={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
