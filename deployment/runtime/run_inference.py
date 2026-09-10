#!/usr/bin/env python3
"""Verify bundle integrity before starting C++ inference (standard library only).

--check-only never loads plugins or deserializes engines, and is not a GPU test.
"""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deployment.tensorrt.builder import PLUGIN_FILES, load_json, sha256_file


ENGINE_OPTIONS = {
    "camera_bev": "--camera-engine",
    "lidar_raw": "--lidar-engine",
    "fusion_dal": "--fusion-engine",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def is_sha256(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def identity(model, allow_random_init):
    require(isinstance(model, dict), "missing model provenance")
    require(is_sha256(model.get("config_sha256")), "missing/invalid config SHA-256")
    diagnostic = model.get("random_init_diagnostic")
    require(type(diagnostic) is bool, "missing random_init_diagnostic flag")
    if diagnostic:
        require(allow_random_init, "random-init bundle requires --allow-random-init")
        require(type(model.get("seed")) is int, "diagnostic model is missing seed")
        require(model.get("checkpoint_sha256") is None, "diagnostic checkpoint mismatch")
    else:
        require(is_sha256(model.get("checkpoint_sha256")),
                "missing/invalid checkpoint SHA-256")
    return tuple(model.get(key) for key in (
        "config_sha256", "checkpoint_sha256", "random_init_diagnostic", "seed"
    ))


def checked_file(directory, recorded_path, digest):
    # Build manifests record absolute paths. Resolve the filename beside this
    # bundle instead, so host/container mount paths can change on deployment.
    require(isinstance(recorded_path, str) and recorded_path, "missing artifact path")
    name = Path(recorded_path).name
    require(name not in ("", ".", ".."), "invalid artifact filename")
    require(is_sha256(digest), "missing/invalid SHA-256 for " + name)
    path = directory / name
    require(path.is_file(), "artifact does not exist: " + str(path))
    require(path.stat().st_size > 0, "empty artifact: " + str(path))
    require(sha256_file(path) == digest, "SHA-256 mismatch: " + str(path))
    return path.resolve()


def verify_bundle(bundle_path, points=None, allow_random_init=False):
    bundle_path = Path(bundle_path).expanduser().resolve()
    root = bundle_path.parent
    bundle = load_json(bundle_path, "engine bundle manifest")
    require(bundle.get("schema_version") == 2, "expected bundle schema_version=2")
    expected_identity = identity(bundle.get("model"), allow_random_init)
    require(bundle.get("precision") in ("fp16", "fp32"), "invalid bundle precision")
    engines = bundle.get("engines")
    require(isinstance(engines, dict) and set(engines) == set(ENGINE_OPTIONS),
            "bundle must contain camera_bev, lidar_raw and fusion_dal")
    resolved, manifests = {}, {}
    targets = set()
    for name in ENGINE_OPTIONS:
        spec = engines[name]
        require(isinstance(spec, dict), "invalid engine entry: " + name)
        manifest_path = checked_file(root, spec.get("build_manifest"),
                                     spec.get("build_manifest_sha256"))
        manifest = load_json(manifest_path, name + " build manifest")
        require(manifest.get("schema_version") == 1, "unsupported build manifest: " + name)
        require(manifest.get("engine_name") == name, "engine name mismatch: " + name)
        require(identity(manifest.get("model"), allow_random_init) == expected_identity,
                "model identity mismatch: " + name)
        require(manifest.get("precision") == bundle["precision"], "precision mismatch: " + name)
        require(manifest.get("engine_sha256") == spec.get("sha256"),
                "engine hash disagreement: " + name)
        engine_path = spec.get("path")
        require(isinstance(engine_path, str) and isinstance(manifest.get("engine"), str),
                "missing engine path: " + name)
        require(Path(engine_path).name == Path(manifest["engine"]).name,
                "engine filename disagreement: " + name)
        resolved[name] = str(checked_file(root, engine_path, spec.get("sha256")))
        target = (manifest.get("tensorrt_version"), manifest.get("machine"))
        require(all(isinstance(value, str) and value for value in target),
                "missing build target: " + name)
        targets.add(target)
        manifests[name] = manifest
    require(len(targets) == 1, "engines were built for different TensorRT versions/machines")

    # The current A/C builders emit no custom plugins.
    for name in ("camera_bev", "fusion_dal"):
        require(manifests[name].get("plugins") == [], "unsupported extra plugins on " + name)
    plugin_specs = manifests["lidar_raw"].get("plugins")
    require(isinstance(plugin_specs, list) and len(plugin_specs) == len(PLUGIN_FILES),
            "LiDAR manifest must contain exactly five plugins")
    plugin_paths = {}
    for spec in plugin_specs:
        require(isinstance(spec, dict) and isinstance(spec.get("path"), str),
                "invalid plugin entry")
        name = Path(spec["path"]).name
        require(name in PLUGIN_FILES and name not in plugin_paths, "unknown/duplicate plugin: " + name)
        plugin_paths[name] = str(checked_file(root / "plugins", spec["path"], spec.get("sha256")))

    lidar = manifests["lidar_raw"]
    profile = lidar.get("point_profile")
    require(isinstance(profile, dict), "missing point profile")
    bounds = []
    for key in ("min", "opt", "max"):
        shape = profile.get(key)
        require(isinstance(shape, list) and len(shape) == 2 and shape[1] == 5
                and type(shape[0]) is int, "invalid points profile: " + key)
        bounds.append(shape[0])
    minimum, optimum, maximum = bounds
    capacity = lidar.get("capacity")
    require(isinstance(capacity, dict), "missing LiDAR capacity")
    for key in ("max_points", "max_pillars", "max_sets_per_shift"):
        require(type(capacity.get(key)) is int and capacity[key] > 0,
                "missing/invalid LiDAR capacity: " + key)
    require(0 < minimum <= optimum <= maximum <= capacity["max_points"],
            "point profile exceeds capacity or is unordered")
    selected_points = optimum if points is None else points
    require(type(selected_points) is int and minimum <= selected_points <= maximum,
            f"points={selected_points} outside Engine B profile [{minimum}, {maximum}]")
    trt_version, machine = next(iter(targets))
    return {
        "check": "artifact_integrity_only",
        "bundle": str(bundle_path),
        "precision": bundle["precision"],
        "model": bundle["model"],
        "engines": resolved,
        "plugins": [plugin_paths[name] for name in PLUGIN_FILES],
        "points": selected_points,
        "point_profile": profile,
        "capacity": capacity,
        "build_target": {"tensorrt_version": trt_version, "machine": machine},
    }


def runtime_command(args, verified):
    command = [str(args.runtime.expanduser().resolve())]
    for name, option in ENGINE_OPTIONS.items():
        command.extend((option, verified["engines"][name]))
    for plugin in verified["plugins"]:
        command.extend(("--plugin", plugin))
    command.extend(("--points", str(verified["points"]), "--warmup", str(args.warmup),
                    "--iterations", str(args.iterations)))
    for name in ("latency", "memory", "profile", "single_thread_submit",
                 "synthetic_unique_pillars"):
        if getattr(args, name):
            command.append("--" + name.replace("_", "-"))
    if args.memory_sample_ms is not None:
        command.extend(("--memory-sample-ms", str(args.memory_sample_ms)))
    return command


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--runtime", type=Path,
                        default=REPO_ROOT / "deployment/artifacts/runtime-build/bevfusion_trt_infer")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument("--points", type=int, help="Default: bundle point-opt")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--memory-sample-ms", type=int)
    for flag in ("latency", "memory", "profile", "single-thread-submit", "synthetic-unique-pillars"):
        parser.add_argument("--" + flag, action="store_true")
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("warmup must be >= 0 and iterations must be > 0")
    if args.memory_sample_ms is not None and args.memory_sample_ms <= 0:
        parser.error("memory-sample-ms must be > 0")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        verified = verify_bundle(args.bundle, args.points, args.allow_random_init)
        print(json.dumps(verified, indent=2, sort_keys=True), flush=True)
        if args.check_only:
            return 0
        command = runtime_command(args, verified)
        require(Path(command[0]).is_file() and os.access(command[0], os.X_OK),
                "runtime executable missing/not executable: " + command[0])
        # No shell or unverified engine/plugin overrides are accepted.
        return subprocess.run(command, check=False).returncode
    except (OSError, ValueError, TypeError) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
