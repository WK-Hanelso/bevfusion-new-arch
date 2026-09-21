"""Pure-PyYAML tests for the Phase 2 recursive config family."""

import itertools
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]
ABLATION_DIR = ROOT / "configs/nuscenes/det/ablation"
REFERENCE = (
    ROOT
    / "configs/nuscenes/det/transfusion/secfpn/lidar"
    / "dsvt_dgf_dal_widthformer_0p3.yaml"
)

CANONICAL_LEAVES = [
    f"dsvt{dsvt}_wf{wf}_gf{gf}_dal{dal}.yaml"
    for dsvt, wf, gf, dal in itertools.product((0, 1), repeat=4)
]
ALIASES = {
    "b0_legacy.yaml": "dsvt0_wf0_gf0_dal0.yaml",
    "e1_dsvt.yaml": "dsvt1_wf0_gf0_dal0.yaml",
    "e2_widthformer.yaml": "dsvt0_wf1_gf0_dal0.yaml",
    "e3_gfusion.yaml": "dsvt0_wf0_gf1_dal0.yaml",
    "e4_dal.yaml": "dsvt0_wf0_gf0_dal1.yaml",
    "c2_dsvt_widthformer.yaml": "dsvt1_wf1_gf0_dal0.yaml",
    "c3_dsvt_widthformer_gfusion.yaml": "dsvt1_wf1_gf1_dal0.yaml",
    "c4_full.yaml": "dsvt1_wf1_gf1_dal1.yaml",
}
LEAVES = CANONICAL_LEAVES + list(ALIASES)


def _deep_merge(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _recursive_paths(path):
    paths = [path]
    parent = path.parent
    while parent != parent.parent:
        default = parent / "default.yaml"
        if default != path and default.is_file():
            paths.append(default)
        if parent == ROOT:
            break
        parent = parent.parent
    return list(reversed(paths))


def _load_recursive(path):
    resolved = {}
    for config_path in _recursive_paths(path):
        document = yaml.safe_load(config_path.read_text()) or {}
        _deep_merge(resolved, document)
    return resolved


@pytest.mark.parametrize(
    "leaf", ["dsvt1_wf1_gf1_dal1.yaml", "c4_full.yaml"]
)
def test_full_model_blocks_match_existing_full_config(leaf):
    candidate = _load_recursive(ABLATION_DIR / leaf)["model"]
    reference = _load_recursive(REFERENCE)["model"]
    for block in ("encoders", "fuser", "heads", "decoder"):
        assert candidate[block] == reference[block]


def test_generated_leaf_set_is_complete():
    actual = {
        path.name for path in ABLATION_DIR.glob("*.yaml") if path.name != "default.yaml"
    }
    actual = {name for name in actual if not (name.startswith('s1_') or name.endswith(('_perf.yaml', '_perf_noaug.yaml')))}  # stage configs are hand-written
    assert actual == set(LEAVES)


@pytest.mark.parametrize("alias,canonical", ALIASES.items())
def test_alias_matches_canonical_content_and_names_it_in_header(alias, canonical):
    alias_text = (ABLATION_DIR / alias).read_text()
    canonical_text = (ABLATION_DIR / canonical).read_text()
    assert yaml.safe_load(alias_text) == yaml.safe_load(canonical_text)
    assert f"# Canonical: {canonical}" in alias_text.splitlines()[:4]


@pytest.mark.parametrize("leaf", LEAVES)
def test_common_experiment_contract(leaf):
    config = _load_recursive(ABLATION_DIR / leaf)
    pipeline_types = [stage["type"] for stage in config["train_pipeline"]]
    assert config["model"]["type"] == "ModularBEVFusion"
    assert config["camera_names"] == [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]
    assert config["image_size"] == [256, 704]
    assert config["lidar_sweeps"] == 0
    assert config["max_epochs"] == 20
    assert config["data"]["train"]["type"] == "CBGSDataset"
    assert "LoadPointsFromMultiSweeps" not in pipeline_types
    assert "ObjectPaste" not in pipeline_types
    assert config["load_from"] is None


@pytest.mark.parametrize("leaf", LEAVES)
def test_leaf_selects_expected_modules_and_geometry(leaf):
    canonical = ALIASES.get(leaf, leaf)
    dsvt = canonical[4] == "1"
    widthformer = canonical[8] == "1"
    gfusion = canonical[12] == "1"
    dal = canonical[17] == "1"
    factor = 2 if dsvt else 8
    grid_size = [360, 360, 1] if dsvt else [1440, 1440, 41]
    voxel_size = [0.3, 0.3, 8.0] if dsvt else [0.075, 0.075, 0.2]

    config = _load_recursive(ABLATION_DIR / leaf)
    model = config["model"]
    lidar = model["encoders"]["lidar"]
    if isinstance(lidar, str):
        assert lidar == "${lidar_legacy}"
        lidar = config["lidar_legacy"]
    resolved_lidar_type = (
        lidar["type"] if "type" in lidar else lidar["backbone"]["type"]
    )
    assert resolved_lidar_type == ("DSVTLidarEncoder" if dsvt else "SparseEncoder")
    expected_vtransform = "WidthFormerTransform" if widthformer else "DepthLSSTransform"
    assert model["encoders"]["camera"]["vtransform"]["type"] == expected_vtransform
    assert model["fuser"]["type"] == ("DepthGFusion" if gfusion else "ConvFuser")
    assert model["heads"]["object"]["type"] == (
        "DALDecoupledHead" if dal else "TransFusionHead"
    )
    assert config["voxel_size"] == voxel_size
    assert model["heads"]["object"]["train_cfg"]["grid_size"] == grid_size
    assert model["heads"]["object"]["test_cfg"]["grid_size"] == grid_size
    assert model["heads"]["object"]["train_cfg"]["out_size_factor"] == factor
    assert model["heads"]["object"]["test_cfg"]["out_size_factor"] == factor
    assert model["heads"]["object"]["bbox_coder"]["out_size_factor"] == factor


def test_leaf_files_do_not_set_load_from():
    for leaf in LEAVES:
        document = yaml.safe_load((ABLATION_DIR / leaf).read_text()) or {}
        assert "load_from" not in document
