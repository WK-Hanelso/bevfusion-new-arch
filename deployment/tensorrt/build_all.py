#!/usr/bin/env python3
"""Build the production camera, raw-LiDAR, and fusion TensorRT bundle."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deployment.tensorrt.builder import (
    BuildOptions,
    build_engine,
    load_json,
    sha256_file,
)
from deployment.tensorrt.lidar_builder import (
    build_raw_lidar_engine,
    load_lidar_artifacts,
)


def static_artifact_paths(directory: Path, name: str):
    return directory / f"{name}.onnx", directory / f"{name}.manifest.json"


def model_identity(model):
    return (
        model.get("config_sha256"),
        model.get("checkpoint_sha256"),
        model.get("random_init_diagnostic"),
        model.get("seed"),
    )


def validate_export_bundle(
    onnx_dir: Path, lidar_artifact_dir: Path, allow_random_init: bool
) -> None:
    manifests = []
    for name in ("camera_bev", "fusion_dal"):
        _, manifest_path = static_artifact_paths(onnx_dir, name)
        manifest = load_json(manifest_path, f"{name} export manifest")
        if manifest.get("engine") != name:
            raise ValueError(
                f"{manifest_path} declares engine={manifest.get('engine')}, "
                f"expected={name}"
            )
        manifests.append(manifest)
    _, lidar_manifest, _ = load_lidar_artifacts(
        lidar_artifact_dir, allow_random_init
    )
    manifests.append(lidar_manifest)

    identities = []
    for manifest in manifests:
        model = manifest.get("model")
        if not isinstance(model, dict):
            raise ValueError("export manifest is missing model provenance")
        if model.get("random_init_diagnostic") and not allow_random_init:
            raise ValueError("random-init artifacts cannot build a production bundle")
        identities.append(model_identity(model))
    if len(set(identities)) != 1:
        raise ValueError(
            "camera, raw-LiDAR and fusion artifacts do not share one model "
            f"identity: {identities}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-dir", type=Path, required=True)
    parser.add_argument("--lidar-artifact-dir", type=Path)
    parser.add_argument("--plugin-dir", type=Path, required=True)
    parser.add_argument("--engine-dir", type=Path, required=True)
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

    onnx_dir = args.onnx_dir.expanduser().resolve()
    lidar_artifact_dir = (
        args.lidar_artifact_dir.expanduser().resolve()
        if args.lidar_artifact_dir is not None
        else onnx_dir / "lidar_trt"
    )
    plugin_dir = args.plugin_dir.expanduser().resolve()
    engine_dir = args.engine_dir.expanduser().resolve()
    validate_export_bundle(
        onnx_dir, lidar_artifact_dir, args.allow_random_init
    )
    timing_cache = (
        args.timing_cache.expanduser().resolve()
        if args.timing_cache is not None
        else engine_dir / "timing.cache"
    )
    options = BuildOptions(
        precision=args.precision,
        workspace_gib=args.workspace_gib,
        allow_tf32=args.allow_tf32,
        optimization_level=args.optimization_level,
        max_aux_streams=args.max_aux_streams,
        timing_cache=timing_cache,
        allow_random_init=args.allow_random_init,
    )

    built = {}
    camera_onnx, camera_export_manifest = static_artifact_paths(
        onnx_dir, "camera_bev"
    )
    camera_build_manifest = (
        engine_dir / f"camera_bev_{args.precision}.manifest.json"
    )
    built["camera_bev"] = build_engine(
        "camera_bev",
        camera_onnx,
        camera_export_manifest,
        engine_dir / f"camera_bev_{args.precision}.engine",
        camera_build_manifest,
        options,
    )
    print(
        f"PASS camera_bev sha256={built['camera_bev']['engine_sha256']}"
    )

    lidar_build_manifest = (
        engine_dir / f"dsvt_lidar_{args.precision}.manifest.json"
    )
    built["lidar_raw"] = build_raw_lidar_engine(
        lidar_artifact_dir,
        plugin_dir,
        engine_dir / f"dsvt_lidar_{args.precision}.engine",
        lidar_build_manifest,
        (args.point_min, args.point_opt, args.point_max),
        options,
    )
    print(f"PASS lidar_raw sha256={built['lidar_raw']['engine_sha256']}")

    onnx_path, export_manifest = static_artifact_paths(onnx_dir, "fusion_dal")
    fusion_build_manifest = (
        engine_dir / f"fusion_dal_{args.precision}.manifest.json"
    )
    built["fusion_dal"] = build_engine(
        "fusion_dal",
        onnx_path,
        export_manifest,
        engine_dir / f"fusion_dal_{args.precision}.engine",
        fusion_build_manifest,
        options,
    )
    print(
        f"PASS fusion_dal sha256={built['fusion_dal']['engine_sha256']}"
    )

    manifest_paths = {
        "camera_bev": camera_build_manifest,
        "lidar_raw": lidar_build_manifest,
        "fusion_dal": fusion_build_manifest,
    }
    bundle_manifest = {
        "schema_version": 2,
        "precision": args.precision,
        "model": built["camera_bev"]["model"],
        "public_abi": {
            "camera": {
                "inputs": ["images", "geometry"],
                "outputs": ["camera_bev"],
            },
            "lidar": {
                "inputs": ["points"],
                "outputs": ["lidar_bev", "lidar_status"],
            },
            "fusion": {
                "inputs": ["camera_bev", "lidar_bev"],
                "outputs": ["boxes", "scores", "labels"],
            },
        },
        "engines": {
            name: {
                "path": built[name]["engine"],
                "sha256": built[name]["engine_sha256"],
                "build_manifest": str(manifest_paths[name]),
                "build_manifest_sha256": sha256_file(manifest_paths[name]),
            }
            for name in ("camera_bev", "lidar_raw", "fusion_dal")
        },
    }
    bundle_path = engine_dir / f"bundle_{args.precision}.manifest.json"
    bundle_path.write_text(
        json.dumps(bundle_manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"PASS bundle_manifest={bundle_path}")


if __name__ == "__main__":
    main()
