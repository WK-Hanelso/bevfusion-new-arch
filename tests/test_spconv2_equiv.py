"""CPU compatibility tests and an optional GPU backend equivalence test."""

import importlib.util
from collections import OrderedDict
from pathlib import Path

import pytest
import torch
from torch import nn


def _load_compat_module():
    """Load the pure compatibility module without importing mmdet3d/mmcv."""

    path = (
        Path(__file__).resolve().parents[1]
        / "mmdet3d"
        / "ops"
        / "spconv_compat.py"
    )
    spec = importlib.util.spec_from_file_location("bevfusion_spconv_compat", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compat = _load_compat_module()


def test_backend_resolution_defaults_to_legacy(monkeypatch):
    monkeypatch.delenv("BEVFUSION_SPCONV", raising=False)
    assert compat.resolve_spconv_backend() == "legacy"
    monkeypatch.setenv("BEVFUSION_SPCONV", "v2")
    assert compat.resolve_spconv_backend() == "v2"
    assert compat.resolve_spconv_backend("legacy") == "legacy"


def test_spconv_weight_layout_round_trip():
    legacy = torch.arange(3 * 2 * 1 * 4 * 5).reshape(3, 2, 1, 4, 5)
    v2 = compat.legacy_to_spconv2_weight(legacy)
    assert v2.shape == (5, 3, 2, 1, 4)
    torch.testing.assert_close(compat.spconv2_to_legacy_weight(v2), legacy)


def test_state_dict_keys_are_preserved_and_conv_weight_is_converted():
    legacy_weight = torch.randn(3, 3, 3, 4, 8)
    source = OrderedDict(
        [
            ("conv_input.0.weight", legacy_weight),
            ("conv_input.1.running_mean", torch.randn(8)),
            ("unexpected", torch.tensor(1.0)),
        ]
    )
    target = OrderedDict(
        [
            ("conv_input.0.weight", torch.empty(8, 3, 3, 3, 4)),
            ("conv_input.1.running_mean", torch.empty(8)),
        ]
    )

    converted = compat.convert_spconv_state_dict(source, target)

    assert list(converted) == list(source)
    assert compat.map_spconv_state_dict_key("conv_input.0.weight", set(target)) == (
        "conv_input.0.weight"
    )
    assert compat.map_spconv_state_dict_key("unexpected", set(target)) is None
    torch.testing.assert_close(
        converted["conv_input.0.weight"],
        compat.legacy_to_spconv2_weight(legacy_weight),
    )
    torch.testing.assert_close(
        converted["conv_input.1.running_mean"], source["conv_input.1.running_mean"]
    )
    assert converted["unexpected"] is source["unexpected"]


def test_recursive_load_hook_accepts_legacy_weight_shape():
    class V2LikeConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(8, 3, 3, 3, 4))

    module = V2LikeConv()
    compat.register_legacy_weight_load_hook(module)
    legacy_weight = torch.randn(3, 3, 3, 4, 8)

    module.load_state_dict({"weight": legacy_weight})

    torch.testing.assert_close(
        module.weight, compat.legacy_to_spconv2_weight(legacy_weight)
    )


def _require_gpu_backends():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for sparse convolution equivalence")
    try:
        import spconv.pytorch  # noqa: F401
    except ImportError:
        pytest.skip("spconv.pytorch is not installed")
    try:
        from mmdet3d.models.backbones.sparse_encoder import SparseEncoder
    except (ImportError, OSError) as exc:
        pytest.skip(f"legacy mmdet3d sparse ops are unavailable: {exc}")
    return SparseEncoder


def test_sparse_encoder_legacy_and_v2_are_equivalent_on_gpu():
    SparseEncoder = _require_gpu_backends()
    torch.manual_seed(7)
    model_kwargs = dict(
        in_channels=5,
        sparse_shape=[17, 32, 32],
        base_channels=8,
        output_channels=16,
        encoder_channels=((8,), (16, 16), (32, 32), (32, 32)),
        encoder_paddings=((1,), (1, 1), (1, 1), ((0, 1, 1), 1)),
        block_type="conv_module",
    )
    legacy = SparseEncoder(**model_kwargs, spconv_backend="legacy").cuda().eval()
    v2 = SparseEncoder(**model_kwargs, spconv_backend="v2").cuda().eval()
    # The v2 convolution pre-hooks convert legacy layouts during recursive load.
    v2.load_state_dict(legacy.state_dict(), strict=True)

    # Unique, sorted coordinates avoid backend-dependent duplicate handling.
    all_indices = torch.randperm(17 * 32 * 32)[:2500].sort().values
    z = torch.div(all_indices, 32 * 32, rounding_mode="floor")
    remainder = all_indices % (32 * 32)
    y = torch.div(remainder, 32, rounding_mode="floor")
    x = remainder % 32
    coors = torch.stack((torch.zeros_like(z), z, y, x), dim=1).int().cuda()
    features = torch.randn(coors.shape[0], 5, device="cuda", dtype=torch.float32)

    with torch.no_grad():
        legacy_output = legacy(features, coors, batch_size=1)
        v2_output = v2(features, coors, batch_size=1)

    assert legacy_output.shape == v2_output.shape
    max_abs_diff = (legacy_output - v2_output).abs().max().item()
    assert max_abs_diff <= 1e-3
