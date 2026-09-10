#!/usr/bin/env python3
"""Build raw points [N,5] to LiDAR BEV TensorRT Engine B."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deployment.tensorrt.builder import BuildOptions
from deployment.tensorrt.lidar_builder import build_raw_lidar_engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--plugin-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-manifest", type=Path)
    parser.add_argument("--point-min", type=int, default=1)
    parser.add_argument("--point-opt", type=int, required=True)
    parser.add_argument("--point-max", type=int, required=True)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--workspace-gib", type=float, default=4.0)
    parser.add_argument("--timing-cache", type=Path)
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--optimization-level", type=int, default=3)
    parser.add_argument("--max-aux-streams", type=int, default=0)
    parser.add_argument("--allow-random-init", action="store_true")
    args = parser.parse_args()

    build_manifest = args.build_manifest or args.output.with_suffix(
        ".manifest.json"
    )
    options = BuildOptions(
        precision=args.precision,
        workspace_gib=args.workspace_gib,
        allow_tf32=args.allow_tf32,
        optimization_level=args.optimization_level,
        max_aux_streams=args.max_aux_streams,
        timing_cache=args.timing_cache,
        allow_random_init=args.allow_random_init,
    )
    manifest = build_raw_lidar_engine(
        args.artifact_dir,
        args.plugin_dir,
        args.output,
        build_manifest,
        (args.point_min, args.point_opt, args.point_max),
        options,
    )
    print(
        f"PASS engine={manifest['engine']} input=points:[N,5] "
        f"outputs=lidar_bev,lidar_status sha256={manifest['engine_sha256']}"
    )


if __name__ == "__main__":
    main()
