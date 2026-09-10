"""Shared model and artifact contract for Thor TensorRT exporters.

Production exporters must derive every engine from one resolved model definition
and checkpoint. Random initialization is supported only as an explicit diagnostic.
"""

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from mmcv import Config

from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / (
    "configs/nuscenes/det/transfusion/secfpn/lidar/"
    "dsvt_dgf_dal_widthformer_0p3.yaml"
)

BEV_RANGE = (-54.0, -54.0, 54.0, 54.0)
BEV_RESOLUTION = 0.6
BEV_SHAPE = (180, 180)
CAMERA_BEV_SHAPE = (1, 80, 180, 180)
LIDAR_BEV_SHAPE = (1, 256, 180, 180)
POINT_CHANNELS = 5


def disable_external_pretrained_init(cfg: Config) -> int:
    """Prevent deployment export from downloading initialization checkpoints.

    A production export restores the complete trained checkpoint immediately
    after model construction. External ``Pretrained`` init_cfg entries are
    therefore both unnecessary and an offline-export failure mode. Diagnostic
    random-init exports intentionally use each module's local initialization.
    """

    disabled = 0

    def visit(node: Any) -> None:
        nonlocal disabled
        if isinstance(node, dict):
            init_cfg = node.get("init_cfg")
            if isinstance(init_cfg, dict) and init_cfg.get("type") == "Pretrained":
                node["init_cfg"] = None
                disabled += 1
            if isinstance(node.get("pretrained"), str):
                node["pretrained"] = None
                disabled += 1
            for value in list(node.values()):
                visit(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                visit(value)

    visit(cfg.model)
    return disabled


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_strict_deployment_checkpoint(model, path, map_location):
    """Load learned state strictly, migrating only one verified legacy buffer."""

    checkpoint = torch.load(str(path), map_location=map_location)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"checkpoint must be a mapping: {path}")
    state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        if not checkpoint or not all(
            isinstance(value, torch.Tensor) for value in checkpoint.values()
        ):
            raise ValueError(f"checkpoint has no state_dict: {path}")
        state_dict = checkpoint
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"checkpoint state_dict must be a mapping: {path}")

    normalized = {}
    for key, value in state_dict.items():
        normalized_key = key[7:] if key.startswith("module.") else key
        if normalized_key in normalized:
            raise ValueError(
                f"duplicate checkpoint key after prefix normalization: {normalized_key}"
            )
        normalized[normalized_key] = value

    migrated = []
    legacy_key = "encoders.camera.vtransform.depth_values"
    if legacy_key in normalized:
        expected = model.encoders["camera"]["vtransform"].depth_values.detach().cpu()
        supplied = normalized[legacy_key].detach().cpu()
        if (
            supplied.shape != expected.shape
            or supplied.dtype != expected.dtype
            or not torch.allclose(supplied, expected, rtol=0.0, atol=1e-6)
        ):
            raise ValueError(
                f"legacy checkpoint buffer does not match config: {legacy_key}"
            )
        del normalized[legacy_key]
        migrated.append(legacy_key)

    model.load_state_dict(normalized, strict=True)
    metadata = checkpoint.get("meta", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise TypeError(f"checkpoint meta must be a mapping: {path}")
    return dict(metadata), migrated


def load_deployment_model(
    config_path: Path,
    checkpoint_path: Optional[Path],
    allow_random_init: bool = False,
    seed: int = 0,
    map_location: str = "cpu",
) -> Tuple[torch.nn.Module, Config, Dict[str, Any]]:
    """Build the complete model and load the one authoritative checkpoint."""

    config_path = Path(config_path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if checkpoint_path is None and not allow_random_init:
        raise ValueError(
            "a checkpoint is required for production export; pass "
            "--allow-random-init only for an explicitly diagnostic artifact"
        )

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if config_path.suffix.lower() == ".json":
        cfg = Config(json.loads(config_path.read_text()), filename=str(config_path))
    else:
        try:
            from torchpack.utils.config import configs
        except ImportError as error:
            raise RuntimeError(
                "recursive YAML loading requires torchpack; run "
                "tools/resolve_deployment_config.py on the training host and "
                "pass its JSON output to the Thor exporter"
            ) from error
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
    disabled_pretrained_initializers = disable_external_pretrained_init(cfg)
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))

    checkpoint_meta: Dict[str, Any] = {}
    checkpoint_migrations = []
    resolved_checkpoint = None
    if checkpoint_path is not None:
        resolved_checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not resolved_checkpoint.is_file():
            raise FileNotFoundError(resolved_checkpoint)
        checkpoint_meta, checkpoint_migrations = load_strict_deployment_checkpoint(
            model, resolved_checkpoint, map_location
        )

    validate_model_contract(model)
    model.eval()
    provenance = {
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "checkpoint": str(resolved_checkpoint) if resolved_checkpoint else None,
        "checkpoint_sha256": (
            sha256_file(resolved_checkpoint) if resolved_checkpoint else None
        ),
        "random_init_diagnostic": resolved_checkpoint is None,
        "seed": seed if resolved_checkpoint is None else None,
        "external_pretrained_initializers_disabled": (
            disabled_pretrained_initializers
        ),
        "checkpoint_migrations": checkpoint_migrations,
        "checkpoint_meta": checkpoint_meta,
    }
    return model, cfg, provenance


def validate_model_contract(model: torch.nn.Module) -> None:
    required_encoders = {"camera", "lidar"}
    missing = required_encoders.difference(model.encoders.keys())
    if missing:
        raise ValueError("deployment model is missing encoders: %s" % sorted(missing))
    if model.fuser.__class__.__name__ != "DepthGFusion":
        raise ValueError("Engine C requires DepthGFusion")
    if model.heads["object"].__class__.__name__ != "DALDecoupledHead":
        raise ValueError("Engine C requires DALDecoupledHead")
    lidar = model.encoders["lidar"]["backbone"]
    if lidar.__class__.__name__ != "DSVTLidarEncoder":
        raise ValueError("Engine B requires DSVTLidarEncoder")
    camera = model.encoders["camera"]
    if camera["backbone"].__class__.__name__ != "ResNet":
        raise ValueError("Engine A requires a ResNet camera backbone")
    if camera["vtransform"].__class__.__name__ != "WidthFormerTransform":
        raise ValueError("this deployment profile requires WidthFormerTransform")


def write_manifest(
    path: Path,
    engine_name: str,
    provenance: Dict[str, Any],
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "schema_version": 1,
        "engine": engine_name,
        "model": provenance,
        "bev_contract": {
            "layout": "BCYX",
            "range_xy": BEV_RANGE,
            "resolution_m": BEV_RESOLUTION,
            "shape_yx": BEV_SHAPE,
        },
        "inputs": inputs,
        "outputs": outputs,
    }
    if extra:
        payload.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    )
