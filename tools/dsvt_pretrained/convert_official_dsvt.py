#!/usr/bin/env python3
"""Convert the official DSVT nuScenes checkpoint for DynamicBEVFusion."""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import torch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "pretrained" / "DSVT_Nuscenes_val.pth"
DEFAULT_OUTPUT = ROOT / "pretrained" / "dsvt_nuscenes_official_lidar.pth"
PREFIX = "encoders.lidar.backbone."
MAPPING_VERSION = 1


def _load_core_module():
    path = ROOT / "mmdet3d" / "models" / "backbones" / "dsvt_core.py"
    spec = importlib.util.spec_from_file_location("dsvt_pretrained_core", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def map_key(key: str) -> Optional[str]:
    """Map one official model_state key to a DSVTLidarEncoder state key."""

    if key.startswith("vfe.pfn_layers."):
        return "vfe.layers." + key[len("vfe.pfn_layers.") :]

    posembed = "backbone_3d.input_layer.posembed_layers.0."
    if key.startswith(posembed):
        rest = key[len(posembed) :]
        parts = rest.split(".")
        if len(parts) >= 5 and parts[2] == "position_embedding_head":
            block, shift, _, layer = parts[:4]
            if layer in {"0", "1", "3"}:
                suffix = ".".join(parts[4:])
                return (
                    f"backbone.input_layer.position_layers.{block}.{shift}."
                    f"layers.{layer}.{suffix}"
                )
        return None

    stage = "backbone_3d.stage_0."
    if key.startswith(stage):
        parts = key[len(stage) :].split(".")
        if len(parts) >= 5 and parts[1] == "encoder_list":
            block, attention = parts[0], parts[2]
            if (
                len(parts) >= 6
                and parts[3] == "win_attn"
                and parts[4] == "self_attn"
            ):
                suffix = ".".join(parts[5:])
                return f"backbone.blocks.{block}.layers.{attention}.attention.{suffix}"
            if parts[3] == "win_attn" and parts[4] in {
                "linear1",
                "linear2",
                "norm1",
                "norm2",
            }:
                suffix = ".".join(parts[4:])
                return f"backbone.blocks.{block}.layers.{attention}.{suffix}"
            if parts[3] == "norm":
                suffix = ".".join(parts[4:])
                return f"backbone.blocks.{block}.layer_norms.{attention}.{suffix}"
        return None

    residual = "backbone_3d.residual_norm_stage_0."
    if key.startswith(residual):
        return "backbone.residual_norms." + key[len(residual) :]

    if key.startswith("backbone_2d.blocks."):
        return "neck.blocks." + key[len("backbone_2d.blocks.") :]
    if key.startswith("backbone_2d.deblocks."):
        return "neck.deblocks." + key[len("backbone_2d.deblocks.") :]
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(input_path: Path, output_path: Path, report_path: Path) -> dict:
    checkpoint = torch.load(str(input_path), map_location="cpu")
    if set(checkpoint) != {"model_state"} or not isinstance(
        checkpoint["model_state"], dict
    ):
        raise ValueError("expected an official checkpoint containing only model_state")
    source_state = checkpoint["model_state"]

    core = _load_core_module()
    encoder = core.DSVTLidarEncoder(
        official_layout=True, zero_feature_channels=[4]
    )
    target_state = {
        PREFIX + key: value.detach().cpu() for key, value in encoder.state_dict().items()
    }

    converted: Dict[str, torch.Tensor] = {}
    matched = []
    unexpected = []
    for source_key, tensor in source_state.items():
        relative_target = map_key(source_key)
        if relative_target is None:
            unexpected.append(source_key)
            continue
        target_key = PREFIX + relative_target
        if target_key not in target_state:
            raise KeyError(f"mapped target does not exist: {source_key} -> {target_key}")
        if tensor.shape != target_state[target_key].shape:
            raise ValueError(
                f"shape mismatch: {source_key} {tuple(tensor.shape)} -> "
                f"{target_key} {tuple(target_state[target_key].shape)}"
            )
        converted[target_key] = tensor.detach().cpu()
        matched.append(
            {
                "source": source_key,
                "target": target_key,
                "shape": list(tensor.shape),
                "elements": tensor.numel(),
            }
        )

    # These are deterministic geometry constants registered by our VFE, not
    # learned official parameters. Including them keeps strict=False missing
    # keys limited to the intentionally random adapter.
    initialized = []
    for relative_key in ("vfe.voxel_size", "vfe.point_cloud_range", "vfe.grid_size"):
        target_key = PREFIX + relative_key
        converted[target_key] = target_state[target_key]
        initialized.append(target_key)

    missing = sorted(set(target_state) - set(converted))
    expected_missing = sorted(
        key for key in target_state if key.startswith(PREFIX + "neck.adapter.")
    )
    if missing != expected_missing:
        raise RuntimeError(f"unexpected target keys left uninitialized: {missing}")

    source_sha256 = _sha256(input_path)
    matched_elements = sum(item["elements"] for item in matched)
    source_elements = sum(tensor.numel() for tensor in source_state.values())
    target_elements = sum(tensor.numel() for tensor in target_state.values())
    report = {
        "mapping_version": MAPPING_VERSION,
        "source": str(input_path),
        "source_sha256": source_sha256,
        "output": str(output_path),
        "prefix": PREFIX,
        "matched": matched,
        "initialized": initialized,
        "missing": missing,
        "unexpected": sorted(unexpected),
        "counts": {
            "matched": len(matched),
            "initialized": len(initialized),
            "missing": len(missing),
            "unexpected": len(unexpected),
        },
        "tensor_elements": {
            "matched_official": matched_elements,
            "initialized_geometry": sum(converted[key].numel() for key in initialized),
            "source_total": source_elements,
            "target_total": target_elements,
        },
        "ratios": {
            "official_checkpoint_elements": matched_elements / source_elements,
            "target_elements_initialized": sum(
                tensor.numel() for tensor in converted.values()
            )
            / target_elements,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": converted,
            "meta": {
                "source": "Haiyang-W/DSVT DSVT_Nuscenes_val.pth",
                "sha256": source_sha256,
                "mapping_version": MAPPING_VERSION,
            },
        },
        str(output_path),
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--report",
        type=Path,
        help="default: OUTPUT with .report.json replacing .pth",
    )
    parser.add_argument(
        "--include-head",
        action="store_true",
        help="reserved for the separately approved Phase 2 head mapping",
    )
    args = parser.parse_args()
    if args.include_head:
        parser.error("--include-head belongs to SPEC Phase 2 and is not implemented")
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    report_path = (
        args.report.expanduser().resolve()
        if args.report is not None
        else output_path.with_suffix(".report.json")
    )
    if not input_path.is_file():
        parser.error(f"input checkpoint not found: {input_path}")
    report = convert(input_path, output_path, report_path)
    print(
        "converted "
        f"matched={report['counts']['matched']} "
        f"missing={report['counts']['missing']} "
        f"unexpected={report['counts']['unexpected']} "
        f"elements={report['tensor_elements']['matched_official']}"
    )
    print(f"checkpoint={output_path}")
    print(f"report={report_path}")


if __name__ == "__main__":
    sys.exit(main())
