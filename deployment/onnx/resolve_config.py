#!/usr/bin/env python3
"""Resolve a recursive Torchpack YAML config into a deployment JSON artifact."""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmcv import Config
from torchpack.utils.config import configs

from mmdet3d.utils import recursive_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    try:
        relative_config = config_path.relative_to(REPO_ROOT)
    except ValueError:
        relative_config = config_path
    original_cwd = Path.cwd()
    try:
        os.chdir(str(REPO_ROOT))
        configs.clear()
        configs.load(str(relative_config), recursive=True)
        cfg = Config(recursive_eval(configs), filename=str(config_path))
    finally:
        os.chdir(str(original_cwd))
    payload = cfg._cfg_dict.to_dict()
    payload["_deployment_source_config"] = str(config_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(f"PASS resolved_config={args.output} source={config_path}")


if __name__ == "__main__":
    main()
