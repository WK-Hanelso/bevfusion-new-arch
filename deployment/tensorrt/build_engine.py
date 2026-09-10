#!/usr/bin/env python3
"""Build one TensorRT engine from an exported ONNX and its manifest."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deployment.tensorrt.builder import BuildOptions, build_engine, load_profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--engine-name",
        required=True,
        choices=("camera_bev", "dsvt_bev", "fusion_dal"),
    )
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--export-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-manifest", type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--workspace-gib", type=float, default=4.0)
    parser.add_argument("--timing-cache", type=Path)
    parser.add_argument("--plugin", type=Path, action="append", default=[])
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
        plugins=tuple(args.plugin),
        allow_random_init=args.allow_random_init,
    )
    manifest = build_engine(
        args.engine_name,
        args.onnx,
        args.export_manifest,
        args.output,
        build_manifest,
        options,
        load_profile(args.profile),
    )
    print(
        f"PASS engine={manifest['engine']} "
        f"sha256={manifest['engine_sha256']} "
        f"manifest={Path(build_manifest).expanduser().resolve()}"
    )


if __name__ == "__main__":
    main()
