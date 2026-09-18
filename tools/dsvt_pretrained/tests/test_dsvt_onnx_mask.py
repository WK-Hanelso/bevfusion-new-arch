"""CPU regressions for the TensorRT-compatible DSVT mask export."""

import importlib.util
from pathlib import Path
from unittest import mock

import pytest
import torch


onnx = pytest.importorskip("onnx")
ROOT = Path(__file__).resolve().parents[3]


def _load_module(name, relative_path):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CORE = _load_module("tested_dsvt_mask_core", "mmdet3d/models/backbones/dsvt_core.py")
EXPORT = _load_module("tested_dsvt_mask_export", "deployment/onnx/dsvt_backbone.py")


def _topology():
    indices = torch.tensor([[0, 1, 2], [2, 3, 3]])
    mask = torch.tensor([[0, 0, 0], [0, 0, 1]], dtype=torch.int32)
    return indices, mask, CORE.official_occurrence_gather(indices)


def test_export_attention_matches_eager_multihead_attention():
    torch.manual_seed(61)
    attention = CORE.SetAttention(
        channels=8, heads=2, feedforward=16, official_layout=True
    ).eval()
    features = torch.randn(4, 8)
    positions = torch.randn(4, 8)
    indices, mask, gather = _topology()

    with torch.no_grad():
        expected = attention.forward_with_gather(
            features, indices, mask.bool(), positions, gather
        )
        with mock.patch.object(
            torch.onnx, "is_in_onnx_export", return_value=True
        ):
            actual = attention.forward_with_gather(
                features, indices, mask, positions, gather
            )

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_deployment_backbone_onnx_has_no_cast_to_bool(tmp_path):
    torch.manual_seed(67)
    backbone = CORE.DSVTBackbone(
        channels=8,
        set_size=3,
        block_count=2,
        official_layout=True,
    ).eval()
    wrapper = EXPORT.DSVTDeployWrapper(backbone).eval()
    indices, mask, gather = _topology()
    indices0 = torch.stack((indices, indices))
    indices1 = torch.stack((indices.flip(0), indices.flip(0)))
    masks0 = torch.stack((mask, mask))
    masks1 = torch.stack((mask.flip(0), mask.flip(0)))
    gather0 = torch.stack((gather, gather))
    gather1 = torch.stack(
        tuple(
            CORE.official_occurrence_gather(indices1[axis])
            for axis in range(2)
        )
    )
    inputs = (
        torch.randn(4, 8),
        indices0,
        indices1,
        masks0,
        masks1,
        gather0,
        gather1,
        torch.randn(2, 2, 4, 8),
    )
    output = tmp_path / "dsvt_backbone.onnx"

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            inputs,
            str(output),
            input_names=(
                "src",
                "set_indices_shift_0",
                "set_indices_shift_1",
                "set_masks_shift_0",
                "set_masks_shift_1",
                "gather_shift_0",
                "gather_shift_1",
                "position_embeddings",
            ),
            output_names=("output",),
            opset_version=16,
            do_constant_folding=True,
        )

    graph = onnx.load(str(output))
    onnx.checker.check_model(graph)
    bool_casts = [
        node.name
        for node in graph.graph.node
        if node.op_type == "Cast"
        and any(
            attribute.name == "to"
            and attribute.i == onnx.TensorProto.BOOL
            for attribute in node.attribute
        )
    ]
    input_types = {
        value.name: value.type.tensor_type.elem_type for value in graph.graph.input
    }

    assert bool_casts == []
    assert input_types["set_masks_shift_0"] == onnx.TensorProto.INT32
    assert input_types["set_masks_shift_1"] == onnx.TensorProto.INT32
    # Arithmetic masking (int32 mask -> float, scaled add): no If, no comparison needed.
    assert not any(node.op_type == "If" for node in graph.graph.node)
    assert any(node.op_type == "Cast" for node in graph.graph.node)
