#!/usr/bin/env python3
"""Export the DSVT transformer backbone with topology precomputed outside ONNX."""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn


def load_dsvt_core():
    path = (
        Path(__file__).resolve().parents[2]
        / "mmdet3d/models/backbones/dsvt_core.py"
    )
    spec = importlib.util.spec_from_file_location("dsvt_core_deploy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DSVT_CORE = load_dsvt_core()


class DSVTDeployWrapper(nn.Module):
    """Official-style TensorRT boundary: DSVT blocks without InputLayer."""

    def __init__(self, backbone):
        super().__init__()
        self.blocks = backbone.blocks
        self.residual_norms = backbone.residual_norms

    def forward(
        self,
        src,
        set_indices_shift_0,
        set_indices_shift_1,
        set_masks_shift_0,
        set_masks_shift_1,
        gather_shift_0,
        gather_shift_1,
        position_embeddings,
    ):
        output = src
        indices = (set_indices_shift_0, set_indices_shift_1)
        masks = (set_masks_shift_0, set_masks_shift_1)
        gathers = (gather_shift_0, gather_shift_1)
        for block_id, (block, norm) in enumerate(
            zip(self.blocks, self.residual_norms)
        ):
            residual = output
            shift = block_id % 2
            for axis, layer in enumerate(block.layers):
                output = layer.forward_with_gather(
                    output,
                    indices[shift][axis],
                    masks[shift][axis],
                    position_embeddings[block_id][axis],
                    gathers[shift][axis],
                )
            output = norm(output + residual)
        return output


def make_inputs(backbone, pillars, device):
    if not 1 <= pillars <= 360 * 360:
        raise ValueError(f"pillars must be in [1, 129600], got {pillars}")
    linear = torch.randperm(360 * 360, device=device)[:pillars]
    coords = torch.stack(
        (
            torch.zeros_like(linear),
            torch.zeros_like(linear),
            torch.div(linear, 360, rounding_mode="floor"),
            linear % 360,
        ),
        dim=1,
    ).int()
    src = torch.randn(pillars, 128, device=device)
    _, indices, masks, positions = backbone.input_layer(src, coords)
    gathers = [
        torch.stack(
            tuple(DSVT_CORE.last_occurrence_gather(value) for value in shifted),
            dim=0,
        )
        for shifted in indices
    ]
    position_embeddings = torch.stack(
        tuple(torch.stack(tuple(value), dim=0) for value in positions), dim=0
    )
    return (
        src,
        indices[0],
        indices[1],
        masks[0],
        masks[1],
        gathers[0],
        gathers[1],
        position_embeddings,
    ), coords


def save_reference(directory, names, inputs, output):
    directory.mkdir(parents=True, exist_ok=True)
    for name, tensor in zip((*names, "output_pytorch"), (*inputs, output)):
        array = tensor.detach().cpu().numpy()
        array.tofile(directory / f"{name}.bin")
        np.save(directory / f"{name}.npy", array)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--pillars", type=int, default=5200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opset", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this deployment export")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    backbone = DSVT_CORE.DSVTBackbone().to(device).eval()
    wrapper = DSVTDeployWrapper(backbone).to(device).eval()
    inputs, coords = make_inputs(backbone, args.pillars, device)
    with torch.no_grad():
        expected = backbone(inputs[0], coords)
        actual = wrapper(*inputs)
    maximum_error = (expected - actual).abs().max().item()
    if maximum_error != 0.0:
        raise RuntimeError(f"deployment wrapper mismatch: max_abs={maximum_error}")

    input_names = (
        "src",
        "set_indices_shift_0",
        "set_indices_shift_1",
        "set_masks_shift_0",
        "set_masks_shift_1",
        "gather_shift_0",
        "gather_shift_1",
        "position_embeddings",
    )
    if args.reference_dir is not None:
        save_reference(args.reference_dir, input_names, inputs, actual)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = {
        "src": {0: "pillar_count"},
        "set_indices_shift_0": {1: "set_count_shift_0"},
        "set_indices_shift_1": {1: "set_count_shift_1"},
        "set_masks_shift_0": {1: "set_count_shift_0"},
        "set_masks_shift_1": {1: "set_count_shift_1"},
        "gather_shift_0": {1: "pillar_count"},
        "gather_shift_1": {1: "pillar_count"},
        "position_embeddings": {2: "pillar_count"},
        "output": {0: "pillar_count"},
    }
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            inputs,
            str(args.output),
            input_names=input_names,
            output_names=("output",),
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            do_constant_folding=True,
        )
    graph = onnx.load(str(args.output))
    onnx.checker.check_model(graph)
    shapes = ",".join(
        f"{name}={tuple(tensor.shape)}"
        for name, tensor in zip(input_names, inputs)
    )
    print(
        f"PASS onnx={args.output} size={args.output.stat().st_size} "
        f"nodes={len(graph.graph.node)} max_abs={maximum_error} {shapes}"
    )


if __name__ == "__main__":
    main()
