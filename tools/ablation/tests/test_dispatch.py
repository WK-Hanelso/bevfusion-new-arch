"""MMCV-free tests for ModularBEVFusion call dispatch."""

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
DISPATCH_PATH = ROOT / "mmdet3d/models/fusion_models/dispatch.py"
SPEC = importlib.util.spec_from_file_location("modular_dispatch", DISPATCH_PATH)
DISPATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DISPATCH)


class FakeFuser:
    def __init__(self, input_style):
        self.input_style = input_style
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return "fused"


class FakeHead:
    def __init__(self, needs_lidar_bev):
        self.needs_lidar_bev = needs_lidar_bev
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        return "predictions"


@pytest.mark.parametrize("input_style", ["list", "named"])
@pytest.mark.parametrize("needs_lidar_bev", [False, True])
@pytest.mark.parametrize("modality", ["fusion", "lidar-only", "camera-only"])
def test_four_dispatch_combinations_across_modalities(
    input_style, needs_lidar_bev, modality
):
    features = {
        "fusion": {"camera": "camera", "lidar": "lidar"},
        "lidar-only": {"lidar": "lidar"},
        "camera-only": {"camera": "camera"},
    }[modality]
    fuser = FakeFuser(input_style)
    fused = DISPATCH.select_fuser_call(fuser, features)

    if modality == "fusion":
        assert fused == "fused"
        if input_style == "list":
            assert fuser.calls == [((["camera", "lidar"],), {})]
        else:
            assert fuser.calls == [
                ((), {"camera_bev": "camera", "lidar_bev": "lidar"})
            ]
    else:
        assert fused == next(iter(features.values()))
        assert fuser.calls == []

    head = FakeHead(needs_lidar_bev)
    result = DISPATCH.select_head_call(head, "decoded", features, "metas")
    assert result == "predictions"
    if needs_lidar_bev:
        regression_input = "lidar" if "lidar" in features else "decoded"
        assert head.calls == [("decoded", regression_input, "metas")]
    else:
        assert head.calls == [("decoded", "metas")]


def test_default_fuser_style_is_list():
    fuser = FakeFuser("list")
    del fuser.input_style
    DISPATCH.select_fuser_call(fuser, {"camera": "camera", "lidar": "lidar"})
    assert fuser.calls == [((["camera", "lidar"],), {})]


def test_multiple_sensors_require_fuser():
    with pytest.raises(ValueError, match="require a fuser"):
        DISPATCH.select_fuser_call(None, {"camera": "camera", "lidar": "lidar"})
