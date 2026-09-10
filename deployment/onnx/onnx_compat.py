"""ONNX symbolics missing from the repository's pinned PyTorch 1.10."""

import math

import onnx
import torch
from torch.onnx import symbolic_registry
from torch.onnx import symbolic_helper
from torch.onnx.symbolic_helper import _maybe_get_const


def _grid_sample_symbolic(
    graph,
    input_tensor,
    grid,
    interpolation_mode,
    padding_mode,
    align_corners,
):
    interpolation_mode = _maybe_get_const(interpolation_mode, "i")
    padding_mode = _maybe_get_const(padding_mode, "i")
    align_corners = _maybe_get_const(align_corners, "b")
    modes = {0: "bilinear", 1: "nearest", 2: "bicubic"}
    padding_modes = {0: "zeros", 1: "border", 2: "reflection"}
    if interpolation_mode not in modes:
        raise ValueError(f"unsupported grid_sample mode: {interpolation_mode}")
    if padding_mode not in padding_modes:
        raise ValueError(f"unsupported grid_sample padding: {padding_mode}")
    return graph.op(
        "GridSample",
        input_tensor,
        grid,
        mode_s=modes[interpolation_mode],
        padding_mode_s=padding_modes[padding_mode],
        align_corners_i=int(align_corners),
    )


def _atan2_symbolic(graph, y, x):
    zero = graph.op(
        "Constant", value_t=torch.tensor(0.0, dtype=torch.float32)
    )
    pi = graph.op(
        "Constant", value_t=torch.tensor(math.pi, dtype=torch.float32)
    )
    half_pi = graph.op(
        "Constant", value_t=torch.tensor(math.pi / 2.0, dtype=torch.float32)
    )
    angle = graph.op("Atan", graph.op("Div", y, x))
    x_positive = graph.op("Greater", x, zero)
    x_negative = graph.op("Less", x, zero)
    y_positive = graph.op("Greater", y, zero)
    y_negative = graph.op("Less", y, zero)
    y_nonnegative = graph.op("Not", y_negative)
    negative_x = graph.op(
        "Where",
        y_nonnegative,
        graph.op("Add", angle, pi),
        graph.op("Sub", angle, pi),
    )
    zero_x = graph.op(
        "Where",
        y_positive,
        half_pi,
        graph.op(
            "Where",
            y_negative,
            graph.op("Neg", half_pi),
            zero,
        ),
    )
    return graph.op(
        "Where",
        x_positive,
        angle,
        graph.op("Where", x_negative, negative_x, zero_x),
    )


def prepare_onnx_export(target_opset):
    """Return the native export opset after registering required symbolics."""
    if target_opset != 16:
        raise ValueError(
            "this deployment profile requires ONNX opset 16 for GridSample"
        )
    native_opset = min(target_opset, symbolic_helper._onnx_main_opset)
    if native_opset == target_opset:
        return native_opset
    symbolic_registry.register_op(
        "grid_sampler", _grid_sample_symbolic, "", native_opset
    )
    symbolic_registry.register_op(
        "atan2", _atan2_symbolic, "", native_opset
    )
    return native_opset


def load_and_check_onnx(path, target_opset):
    """Promote the compatible native graph and run the ONNX checker."""
    graph = onnx.load(str(path), load_external_data=False)
    for opset_import in graph.opset_import:
        if opset_import.domain in ("", "ai.onnx"):
            if opset_import.version > target_opset:
                raise RuntimeError(
                    f"graph opset {opset_import.version} exceeds {target_opset}"
                )
            opset_import.version = target_opset
    onnx.save(graph, str(path))
    graph = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(graph)
    return graph
