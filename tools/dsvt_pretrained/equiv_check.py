#!/usr/bin/env python3
"""Numerically compare the official and BEVFusion DSVT LiDAR paths on CPU."""

import argparse
import importlib.util
import json
import platform
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[2]
TOOL_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = Path("/home/hanelso/data/nuscenes/samples/LIDAR_TOP")
DEFAULT_OFFICIAL = ROOT / "pretrained" / "DSVT_Nuscenes_val.pth"
DEFAULT_CONVERTED = ROOT / "pretrained" / "dsvt_nuscenes_official_lidar.pth"
DEFAULT_REPORT = TOOL_DIR / "equiv_report.json"
PREFIX = "encoders.lidar.backbone."
OFFICIAL_PREFIXES = ("vfe.", "backbone_3d.", "map_to_bev.", "backbone_2d.")
POINT_CLOUD_RANGE = (-54.0, -54.0, -5.0, 54.0, 54.0, 3.0)
TOLERANCE = 1e-3


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load {}".format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_models(official_checkpoint: Path, converted_checkpoint: Path):
    sys.path.insert(0, str(TOOL_DIR))
    try:
        from official_ref import OfficialDSVTLidar
    finally:
        sys.path.pop(0)

    official = OfficialDSVTLidar()
    checkpoint = torch.load(str(official_checkpoint), map_location="cpu")
    if set(checkpoint) != {"model_state"}:
        raise ValueError("official checkpoint must contain only model_state")
    selected = {
        key: value
        for key, value in checkpoint["model_state"].items()
        if key.startswith(OFFICIAL_PREFIXES)
    }
    # No key translation: the vendored modules preserve official names.
    official.load_state_dict(selected, strict=True)

    core = _load_module(
        "dsvt_equiv_core", ROOT / "mmdet3d/models/backbones/dsvt_core.py"
    )
    ours = core.DSVTLidarEncoder(
        in_channels=5,
        point_cloud_range=POINT_CLOUD_RANGE,
        voxel_size=(0.3, 0.3, 8.0),
        d_model=128,
        set_size=90,
        block_count=4,
        window_shape=(30, 30, 1),
        out_channels=256,
        official_layout=True,
        zero_feature_channels=[4],
    )
    converted = torch.load(str(converted_checkpoint), map_location="cpu")
    state = converted["state_dict"]
    if not state or any(not key.startswith(PREFIX) for key in state):
        raise ValueError("converted checkpoint keys do not have the expected prefix")
    relative = {key[len(PREFIX) :]: value for key, value in state.items()}
    incompatible = ours.load_state_dict(relative, strict=False)
    # PyTorch does not report BatchNorm.num_batches_tracked as missing for
    # backward-compatible state-dict loading, although it exists in state_dict().
    adapter_state = sorted(
        key for key in ours.state_dict() if key.startswith("neck.adapter.")
    )
    expected_missing = sorted(
        key
        for key in adapter_state
        if not key.endswith("num_batches_tracked")
    )
    if sorted(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "converted load mismatch: missing={} unexpected={}".format(
                incompatible.missing_keys, incompatible.unexpected_keys
            )
        )
    official.eval()
    ours.eval()
    load_report = {
        "official": {
            "strict": True,
            "selected_tensor_count": len(selected),
            "key_mapping": False,
            "prefixes": list(OFFICIAL_PREFIXES),
        },
        "ours": {
            "strict": False,
            "loaded_tensor_count": len(relative),
            "stripped_prefix": PREFIX,
            "missing": sorted(incompatible.missing_keys),
            "uninitialized_adapter_state": adapter_state,
            "unexpected": sorted(incompatible.unexpected_keys),
        },
    }
    return official, ours, load_report


def _read_points(path: Path) -> Tuple[Tensor, int, int]:
    raw = np.fromfile(str(path), dtype=np.float32)
    if raw.size % 5:
        raise ValueError("{} is not an Nx5 float32 point file".format(path))
    raw = raw.reshape(-1, 5)
    raw_count = int(raw.shape[0])
    raw[:, 4] = 0.0  # timestamp for official; zeroed ring channel for ours
    low = np.asarray(POINT_CLOUD_RANGE[:3], dtype=np.float32)
    high = np.asarray(POINT_CLOUD_RANGE[3:], dtype=np.float32)
    valid = ((raw[:, :3] >= low) & (raw[:, :3] < high)).all(axis=1)
    points = torch.from_numpy(raw[valid].copy())
    return points, raw_count, int(points.shape[0])


def _coord_key(coords: Tensor) -> Tensor:
    coords = coords.long()
    return (
        coords[:, 0] * (360 * 360)
        + coords[:, 1] * (360 * 360)
        + coords[:, 2] * 360
        + coords[:, 3]
    )


def _sort_by_coord(features: Tensor, coords: Tensor) -> Tuple[Tensor, Tensor]:
    keys = _coord_key(coords)
    order = torch.argsort(keys)
    return features[order], keys[order]


def _metrics(reference: Tensor, candidate: Tensor) -> Dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError("shape mismatch: {} vs {}".format(reference.shape, candidate.shape))
    shape = list(reference.shape)
    reference = reference.float().reshape(-1)
    candidate = candidate.float().reshape(-1)
    difference = candidate - reference
    max_abs = float(difference.abs().max().item()) if difference.numel() else 0.0
    ref_norm = torch.linalg.vector_norm(reference)
    diff_norm = torch.linalg.vector_norm(difference)
    relative_l2 = float((diff_norm / ref_norm.clamp_min(1e-12)).item())
    cosine = float(
        torch.nn.functional.cosine_similarity(
            reference.double(), candidate.double(), dim=0, eps=1e-12
        ).item()
    )
    return {
        "max_abs_diff": max_abs,
        "relative_l2_diff": relative_l2,
        "cosine_similarity": cosine,
        "elements": int(reference.numel()),
        "shape": shape,
        "pass": max_abs <= TOLERANCE,
    }


def _official_trace(model, points: Tensor):
    batch = torch.cat((points.new_zeros((points.shape[0], 1)), points), dim=1)
    return model.forward_trace(batch, batch_size=1)


def _ours_trace(model, points: Tensor):
    features, coords, batch_size = model.vfe([points])
    features, indices, masks, positions = model.backbone.input_layer(features, coords)
    vfe = features
    blocks = []
    for block_id, (block, norm) in enumerate(
        zip(model.backbone.blocks, model.backbone.residual_norms)
    ):
        residual = features
        features = norm(
            block(features, indices, masks, positions[block_id], block_id) + residual
        )
        blocks.append(features)
    bev = model.scatter(features, coords, batch_size)
    output = model.neck.forward_features(bev)
    return {
        "coords": coords,
        "vfe": vfe,
        "blocks": blocks,
        "bev": bev,
        "backbone_2d": output,
    }


def _compare_sample(path: Path, official, ours) -> Dict:
    points, raw_count, retained_count = _read_points(path)
    with torch.no_grad():
        official_trace = _official_trace(official, points)
        ours_trace = _ours_trace(ours, points)

    official_vfe, official_keys = _sort_by_coord(
        official_trace["vfe"], official_trace["coords"]
    )
    ours_vfe, ours_keys = _sort_by_coord(ours_trace["vfe"], ours_trace["coords"])
    coordinate_match = bool(
        official_keys.shape == ours_keys.shape and torch.equal(official_keys, ours_keys)
    )
    coordinate_info = {
        "official_count": int(official_keys.numel()),
        "ours_count": int(ours_keys.numel()),
        "set_equal": coordinate_match,
        "official_only": int(
            len(set(official_keys.tolist()) - set(ours_keys.tolist()))
        ),
        "ours_only": int(len(set(ours_keys.tolist()) - set(official_keys.tolist()))),
    }
    if not coordinate_match:
        raise RuntimeError("pillar coordinate sets differ for {}".format(path.name))

    stages = {"vfe": _metrics(official_vfe, ours_vfe)}
    for block_id in range(4):
        official_block, block_keys = _sort_by_coord(
            official_trace["blocks"][block_id], official_trace["coords"]
        )
        ours_block, ours_block_keys = _sort_by_coord(
            ours_trace["blocks"][block_id], ours_trace["coords"]
        )
        if not torch.equal(block_keys, ours_block_keys):
            raise RuntimeError("block coordinate order could not be aligned")
        stages["dsvt_block_{}".format(block_id)] = _metrics(
            official_block, ours_block
        )
    stages["bev"] = _metrics(official_trace["bev"], ours_trace["bev"])
    stages["backbone_2d"] = _metrics(
        official_trace["backbone_2d"], ours_trace["backbone_2d"]
    )
    return {
        "file": str(path),
        "raw_point_count": raw_count,
        "range_filtered_point_count": retained_count,
        "pillar_coordinates": coordinate_info,
        "stages": stages,
    }


def _aggregate(samples: List[Dict]) -> Dict[str, Dict]:
    names = list(samples[0]["stages"])
    result = {}
    for name in names:
        rows = [sample["stages"][name] for sample in samples]
        result[name] = {
            "max_abs_diff": max(row["max_abs_diff"] for row in rows),
            "max_relative_l2_diff": max(row["relative_l2_diff"] for row in rows),
            "min_cosine_similarity": min(row["cosine_similarity"] for row in rows),
            "pass": all(row["pass"] for row in rows),
        }
    return result


def _print_report(report: Dict) -> None:
    print("pillar coordinates")
    print("sample | official | ours | set_equal | official_only | ours_only")
    for sample in report["samples"]:
        coords = sample["pillar_coordinates"]
        print(
            "{} | {} | {} | {} | {} | {}".format(
                Path(sample["file"]).name,
                coords["official_count"],
                coords["ours_count"],
                coords["set_equal"],
                coords["official_only"],
                coords["ours_only"],
            )
        )
    print("\nstage aggregate")
    print("stage | max_abs_diff | max_relative_l2_diff | min_cosine | result")
    for name, row in report["aggregate"].items():
        print(
            "{} | {:.9g} | {:.9g} | {:.12g} | {}".format(
                name,
                row["max_abs_diff"],
                row["max_relative_l2_diff"],
                row["min_cosine_similarity"],
                "PASS" if row["pass"] else "FAIL",
            )
        )
    print("\noverall: {}".format(report["verdict"]["result"]))
    if report["verdict"]["first_mismatch"]:
        print("first mismatch: {}".format(report["verdict"]["first_mismatch"]))
    print("report: {}".format(report["report_path"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--official-checkpoint", type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument("--converted-checkpoint", type=Path, default=DEFAULT_CONVERTED)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    files = sorted(args.data_dir.glob("*.pcd.bin"))[:2]
    if len(files) != 2:
        parser.error("expected at least two .pcd.bin files in {}".format(args.data_dir))
    for path in (args.official_checkpoint, args.converted_checkpoint):
        if not path.is_file():
            parser.error("checkpoint not found: {}".format(path))

    torch.set_num_threads(2)
    torch.manual_seed(0)
    official, ours, load_report = _load_models(
        args.official_checkpoint, args.converted_checkpoint
    )
    samples = [_compare_sample(path, official, ours) for path in files]
    aggregate = _aggregate(samples)
    first_mismatch = next(
        (name for name, row in aggregate.items() if not row["pass"]), None
    )
    coordinates_equal = all(
        sample["pillar_coordinates"]["set_equal"] for sample in samples
    )
    report = {
        "schema_version": 1,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "dtype": "float32",
            "mode": "eval",
            "torch_num_threads": torch.get_num_threads(),
        },
        "tolerance": {
            "max_abs_diff": TOLERANCE,
            "relative_diff_definition": "L2(candidate-reference) / L2(reference)",
        },
        "checkpoint_load": load_report,
        "samples": samples,
        "aggregate": aggregate,
        "verdict": {
            "result": "PASS" if coordinates_equal and first_mismatch is None else "FAIL",
            "pillar_coordinate_sets_equal": coordinates_equal,
            "first_mismatch": first_mismatch,
        },
        "report_path": str(args.report.resolve()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _print_report(report)
    if report["verdict"]["result"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
