#!/usr/bin/env python3
"""Generate all Phase 2 ablation leaf configs deterministically."""

import argparse
import copy
import difflib
import sys
from pathlib import Path

import yaml

try:
    from .experiments import EXPERIMENTS, LEGACY_CONFIG_ALIASES
except ImportError:  # Direct execution: python tools/ablation/gen_configs.py
    from experiments import EXPERIMENTS, LEGACY_CONFIG_ALIASES


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "configs/nuscenes/det/ablation"

LIDAR_BLOCKS = {
    "sparse": None,
    "dsvt": {
        "type": "DSVTLidarEncoder",
        "in_channels": 5,
        "point_cloud_range": "${point_cloud_range}",
        "voxel_size": "${voxel_size}",
        "d_model": 128,
        "set_size": 90,
        "block_count": 4,
        "window_shape": [30, 30, 1],
        "out_channels": 256,
        "official_layout": True,
        "zero_feature_channels": [4],
    },
}

VTRANSFORM_BLOCKS = {
    "depth_lss": None,
    "widthformer": {
        "type": "WidthFormerTransform",
        "in_channels": 256,
        "out_channels": 80,
        "image_size": "${image_size}",
        "feature_size": "${[image_size[0] // 16, image_size[1] // 16]}",
        "xbound": [-54.0, 54.0, 0.6],
        "ybound": [-54.0, 54.0, 0.6],
        "zbound": [-10.0, 10.0, 20.0],
        "dbound": [1.0, 60.0, 1.0],
        "ffn_channels": 320,
        "attention_chunk_size": 0,
        "positional_scale": 10.0,
    },
}

FUSER_BLOCKS = {
    "conv": None,
    "dgf": {
        "type": "DepthGFusion",
        "in_channels": [80, 256],
        "out_channels": 256,
        "embed_channels": 256,
        "patch_hidden_channels": 640,
        "num_heads": 8,
        "patch_size": 5,
        "attention_downsample": 4,
        "xbound": [-54.0, 54.0, 0.6],
        "ybound": [-54.0, 54.0, 0.6],
        "camera_layout": "yx",
        "lidar_layout": "yx",
    },
}

LIDAR_GEOMETRY = {
    "sparse": {
        "voxel_size": [0.075, 0.075, 0.2],
        "grid_size": [1440, 1440, 41],
        "out_size_factor": 8,
    },
    "dsvt": {
        "voxel_size": [0.3, 0.3, 8.0],
        "grid_size": [360, 360, 1],
        "out_size_factor": 2,
    },
}


def _combination(bits):
    dsvt, widthformer, gfusion, dal = bits
    return (
        "dsvt" if dsvt else "sparse",
        "widthformer" if widthformer else "depth_lss",
        "dgf" if gfusion else "conv",
        "dal" if dal else "transfusion",
    )


CANONICAL_COMBINATIONS = {
    experiment.config_name: _combination(tuple(map(int, experiment.bits)))
    for experiment in EXPERIMENTS
}

ALIASES = LEGACY_CONFIG_ALIASES

# Selected via `python gen_configs.py --dsvt-lr-mult {0.1|0|none}`; see _leaf_config.
DSVT_BACKBONE_LR_MULT = 0.1


def _leaf_config(lidar, vtransform, fuser, head):
    geometry = LIDAR_GEOMETRY[lidar]
    config = {"voxel_size": copy.deepcopy(geometry["voxel_size"])}
    model = {}
    encoders = {}

    if lidar == "dsvt":
        encoders["lidar"] = copy.deepcopy(LIDAR_BLOCKS[lidar])
        # Pretrained DSVT fine-tuning: the dense-heatmap head starts at prior
        # 0.5 (TransFusion default), whose huge early gradients wreck a
        # pretrained transformer at the full cyclic lr (B200 2026-09-19:
        # A1/C1/P2 collapsed to background, features exploded x300).
        # Standard remedy: 0.1x lr for the pretrained backbone only.
        # DSVT_BACKBONE_LR_MULT: 0.1 = gentle fine-tune (collapsed on B200),
        # 0.0 = frozen pretrained feature extractor (adapter/fuser/head train),
        # None = same lr as everything else (scratch-style, legacy A6000 recipe).
        if DSVT_BACKBONE_LR_MULT is not None:
            config["optimizer"] = {
                "paramwise_cfg": {
                    "custom_keys": {
                        "encoders.lidar.backbone": {"lr_mult": DSVT_BACKBONE_LR_MULT}
                    }
                }
            }
    if vtransform == "widthformer":
        encoders.setdefault("camera", {})["vtransform"] = copy.deepcopy(
            VTRANSFORM_BLOCKS[vtransform]
        )
    if encoders:
        model["encoders"] = encoders
    if fuser == "dgf":
        model["fuser"] = copy.deepcopy(FUSER_BLOCKS[fuser])

    head_block = {
        "train_cfg": {
            "grid_size": copy.deepcopy(geometry["grid_size"]),
            "out_size_factor": geometry["out_size_factor"],
        },
        "test_cfg": {
            "grid_size": copy.deepcopy(geometry["grid_size"]),
            "out_size_factor": geometry["out_size_factor"],
        },
        "bbox_coder": {"out_size_factor": geometry["out_size_factor"]},
    }
    if head == "dal":
        head_block = {
            "type": "DALDecoupledHead",
            "lidar_in_channels": 256,
            **head_block,
        }
    model.setdefault("heads", {})["object"] = head_block
    if model:
        config["model"] = model
    return config


def _render(canonical_name, combination, alias_name=None):
    lidar, vtransform, fuser, head = combination
    description = (
        f"lidar={lidar}, vtransform={vtransform}, fuser={fuser}, head={head}"
    )
    header_lines = [
        "# Generated by tools/ablation/gen_configs.py; do not edit by hand.",
        f"# Canonical: {canonical_name}",
    ]
    if alias_name is not None:
        header_lines.append(f"# Alias: {alias_name}")
    header_lines.append(f"# Combination: {description}")
    header = "\n".join(header_lines) + "\n"
    body = yaml.safe_dump(
        _leaf_config(*combination),
        sort_keys=False,
        default_flow_style=False,
        width=100,
    )
    return header + body


def _generated_files():
    for canonical_name, combination in CANONICAL_COMBINATIONS.items():
        yield canonical_name, _render(canonical_name, combination)
    for alias_name, canonical_name in ALIASES.items():
        combination = CANONICAL_COMBINATIONS[canonical_name]
        yield alias_name, _render(canonical_name, combination, alias_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if committed leaf files differ from generated bytes",
    )
    parser.add_argument(
        "--dsvt-lr-mult",
        default=None,
        help="DSVT backbone lr multiplier written into dsvt leaves: 0.1 (default), 0 (frozen), none (same lr)",
    )
    args = parser.parse_args()
    if args.dsvt_lr_mult is not None:
        global DSVT_BACKBONE_LR_MULT
        DSVT_BACKBONE_LR_MULT = (
            None if args.dsvt_lr_mult.lower() == "none" else float(args.dsvt_lr_mult)
        )

    failures = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generated = list(_generated_files())
    for name, expected in generated:
        path = OUTPUT_DIR / name
        if args.check:
            actual = path.read_text() if path.is_file() else ""
            if actual != expected:
                failures.append(name)
                sys.stderr.writelines(
                    difflib.unified_diff(
                        actual.splitlines(True),
                        expected.splitlines(True),
                        fromfile=str(path),
                        tofile=f"generated/{name}",
                    )
                )
        else:
            path.write_text(expected)

    if failures:
        parser.error("generated config drift: " + ", ".join(failures))
    action = "checked" if args.check else "generated"
    print(
        f"{action} {len(CANONICAL_COMBINATIONS)} canonical and "
        f"{len(ALIASES)} alias ablation leaf configs"
    )


if __name__ == "__main__":
    main()
