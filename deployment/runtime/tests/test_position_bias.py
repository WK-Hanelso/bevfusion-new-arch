"""CPU regression tests for the export-safe continuous position bias."""

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT / "mmdet3d" / "models" / "fusers" / "depth_guided_attention.py"
)


def _load_continuous_position_bias():
    spec = importlib.util.spec_from_file_location(
        "depth_guided_attention_for_test", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ContinuousPositionBias


def _legacy_forward(module, query_grid, sampled_grid, batch_size):
    """Reference the pre-Round-3 implementation, including dynamic squeeze."""
    query_grid = query_grid.reshape(1, -1, 2)
    sampled_grid = sampled_grid.reshape(batch_size * module.num_heads, -1, 2)
    relative = query_grid[:, :, None] - sampled_grid[:, None]
    relative = torch.sign(relative) * torch.log(relative.abs() + 1.0)
    bias = module.mlp(relative).squeeze(-1)
    return bias.reshape(
        batch_size,
        module.num_heads,
        query_grid.shape[1],
        sampled_grid.shape[1],
    )


def test_position_bias_static_index_matches_legacy_output_and_gradients():
    ContinuousPositionBias = _load_continuous_position_bias()
    torch.manual_seed(7)
    module = ContinuousPositionBias(hidden_channels=12, num_heads=3)
    query_grid = torch.randn(2, 4, 2, dtype=torch.float64, requires_grad=True)
    sampled_grid = torch.randn(6, 5, 2, dtype=torch.float64, requires_grad=True)
    module = module.to(dtype=torch.float64)

    actual = module(query_grid, sampled_grid, batch_size=2)
    expected = _legacy_forward(module, query_grid, sampled_grid, batch_size=2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    gradient_targets = (query_grid, sampled_grid, *module.parameters())
    actual_grads = torch.autograd.grad(actual.square().sum(), gradient_targets)
    expected_grads = torch.autograd.grad(
        expected.square().sum(), gradient_targets
    )
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


def test_position_bias_checkpoint_keys_are_unchanged():
    ContinuousPositionBias = _load_continuous_position_bias()
    module = ContinuousPositionBias(hidden_channels=12, num_heads=3)

    assert tuple(module.state_dict()) == (
        "mlp.0.weight",
        "mlp.0.bias",
        "mlp.2.weight",
        "mlp.2.bias",
        "mlp.4.weight",
        "mlp.4.bias",
    )
