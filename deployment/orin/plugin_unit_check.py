#!/usr/bin/env python3
"""Build and execute one-node TensorRT networks for all Engine B plugins."""

import argparse
import ctypes
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


PLUGIN_FILES = (
    "libdynamic_pillar_decorate.so",
    "libdynamic_scatter_max.so",
    "libtensor_barrier.so",
    "libdsvt_rotated_set.so",
    "libdsvt_dense_scatter.so",
)
MAX_PILLARS = 10000
COORD_CAPACITY = 360 * 360
MAX_SETS = 512
SET_SIZE = 90


def trt_major(trt) -> int:
    return int(str(trt.__version__).split(".", 1)[0])


def get_creator(registry, name: str):
    if hasattr(registry, "get_creator"):
        return registry.get_creator(name, "1", "")
    return registry.get_plugin_creator(name, "1", "")


def create_plugin(trt, registry, name: str):
    creator = get_creator(registry, name)
    if creator is None:
        raise RuntimeError(f"missing plugin creator {name}:1")
    fields = trt.PluginFieldCollection([])
    if trt_major(trt) >= 10:
        plugin = creator.create_plugin(name, fields, trt.TensorRTPhase.BUILD)
    else:
        plugin = creator.create_plugin(name, fields)
    if plugin is None:
        raise RuntimeError(f"failed to create plugin {name}:1")
    return plugin


def add_plugin(trt, network, inputs, plugin):
    if trt_major(trt) >= 10:
        return network.add_plugin_v3(inputs, [], plugin)
    return network.add_plugin_v2(inputs, plugin)


def torch_dtype(trt, dtype):
    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
    }
    if dtype not in mapping:
        raise TypeError(f"unsupported TensorRT output dtype: {dtype}")
    return mapping[dtype]


def build_and_run(
    trt,
    logger,
    registry,
    plugin_name: str,
    inputs: Sequence[Tuple[str, object, np.ndarray]],
) -> List[torch.Tensor]:
    builder = trt.Builder(logger)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    tensors = []
    for name, dtype, values in inputs:
        tensor = network.add_input(name, dtype, tuple(values.shape))
        if tensor is None:
            raise RuntimeError(f"failed to add input {name} for {plugin_name}")
        tensors.append(tensor)
    layer = add_plugin(trt, network, tensors, create_plugin(trt, registry, plugin_name))
    if layer is None:
        raise RuntimeError(f"failed to add plugin layer {plugin_name}")
    for index in range(layer.num_outputs):
        output = layer.get_output(index)
        output.name = f"output_{index}"
        network.mark_output(output)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"failed to build one-node network for {plugin_name}")
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError(f"failed to deserialize one-node engine for {plugin_name}")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError(f"failed to create execution context for {plugin_name}")

    device_inputs: Dict[str, torch.Tensor] = {}
    for name, _, values in inputs:
        device_inputs[name] = torch.as_tensor(values, device="cuda").contiguous()
    device_outputs = []
    for index in range(layer.num_outputs):
        name = f"output_{index}"
        shape = tuple(int(axis) for axis in context.get_tensor_shape(name))
        output = torch.empty(
            shape, dtype=torch_dtype(trt, engine.get_tensor_dtype(name)), device="cuda"
        )
        device_outputs.append(output)

    for name, tensor in device_inputs.items():
        if not context.set_tensor_address(name, int(tensor.data_ptr())):
            raise RuntimeError(f"failed to bind input {name} for {plugin_name}")
    for index, tensor in enumerate(device_outputs):
        if not context.set_tensor_address(f"output_{index}", int(tensor.data_ptr())):
            raise RuntimeError(f"failed to bind output {index} for {plugin_name}")
    stream = torch.cuda.current_stream()
    if not context.execute_async_v3(stream_handle=int(stream.cuda_stream)):
        raise RuntimeError(f"execution failed for {plugin_name}")
    stream.synchronize()
    for index, output in enumerate(device_outputs):
        if output.is_floating_point() and not bool(torch.isfinite(output).all()):
            raise AssertionError(f"{plugin_name} output {index} is not finite")
    return device_outputs


def max_abs(actual: torch.Tensor, expected: np.ndarray) -> float:
    reference = torch.as_tensor(expected, device=actual.device, dtype=actual.dtype)
    return float((actual - reference).abs().max().item()) if actual.numel() else 0.0


def check_decorate(trt, logger, registry) -> float:
    points = np.asarray(
        [
            [-53.85, -53.85, -1.0, 0.1, 0.0],
            [-53.75, -53.80, 0.0, 0.2, 0.0],
            [-53.55, -53.85, 1.0, 0.3, 0.0],
            [0.15, 0.15, -2.0, 0.4, 0.0],
            [54.0, 0.0, 0.0, 0.5, 0.0],
            [0.15, 0.15, 4.0, 0.6, 0.0],
        ],
        dtype=np.float32,
    )
    outputs = build_and_run(
        trt, logger, registry, "DynamicPillarDecorate", [("points", trt.float32, points)]
    )
    keys = []
    for point in points:
        x = int(np.floor((point[0] + 54.0) / 0.3))
        y = int(np.floor((point[1] + 54.0) / 0.3))
        keys.append(x * 360 + y if 0 <= x < 360 and 0 <= y < 360 and -5 <= point[2] < 3 else -1)
    unique = sorted(set(key for key in keys if key >= 0))
    offsets = {key: index for index, key in enumerate(unique)}
    expected_features = np.zeros((len(points), 11), dtype=np.float32)
    expected_inverse = np.full((len(points),), -1, dtype=np.int32)
    for index, (point, key) in enumerate(zip(points, keys)):
        if key < 0:
            continue
        members = points[np.asarray(keys) == key, :3]
        x, y = divmod(key, 360)
        expected_features[index, :5] = point
        expected_features[index, 5:8] = point[:3] - members.mean(axis=0)
        expected_features[index, 8] = point[0] - (x * 0.3 + 0.15 - 54.0)
        expected_features[index, 9] = point[1] - (y * 0.3 + 0.15 - 54.0)
        expected_features[index, 10] = point[2] + 1.0
        expected_inverse[index] = offsets[key]
    expected_coords = np.asarray(
        [[0, 0, key % 360, key // 360] for key in unique], dtype=np.int32
    )
    diff = max_abs(outputs[0], expected_features)
    if diff > 1e-5:
        raise AssertionError(f"DynamicPillarDecorate max_abs_diff={diff}")
    np.testing.assert_array_equal(outputs[1].cpu().numpy(), expected_inverse)
    np.testing.assert_array_equal(outputs[2][: len(unique)].cpu().numpy(), expected_coords)
    np.testing.assert_array_equal(outputs[3].cpu().numpy(), [len(unique)])
    return diff


def check_scatter_max(trt, logger, registry) -> float:
    features = (np.arange(MAX_PILLARS * 4, dtype=np.float32).reshape(MAX_PILLARS, 4) - 7) / 31
    inverse = np.arange(MAX_PILLARS, dtype=np.int32)
    outputs = build_and_run(
        trt,
        logger,
        registry,
        "DynamicScatterMax",
        [("features", trt.float32, features), ("inverse", trt.int32, inverse)],
    )
    diff = max_abs(outputs[0], features)
    if diff != 0.0:
        raise AssertionError(f"DynamicScatterMax max_abs_diff={diff}")
    return diff


def check_barrier(trt, logger, registry) -> float:
    values = np.arange(35, dtype=np.float32).reshape(5, 7) / 9
    output = build_and_run(
        trt, logger, registry, "TensorBarrier", [("input", trt.float32, values)]
    )[0]
    diff = max_abs(output, values)
    if diff != 0.0:
        raise AssertionError(f"TensorBarrier max_abs_diff={diff}")
    return diff


def check_rotated_set(trt, logger, registry) -> float:
    coords = np.zeros((COORD_CAPACITY, 4), dtype=np.int32)
    xy = np.asarray([[0, 0], [1, 0], [0, 1], [2, 2]], dtype=np.int32)
    coords[: len(xy), 2] = xy[:, 1]
    coords[: len(xy), 3] = xy[:, 0]
    count = np.asarray([len(xy)], dtype=np.int32)
    outputs = build_and_run(
        trt,
        logger,
        registry,
        "DSVTRotatedSet",
        [("coords", trt.int32, coords), ("pillar_count", trt.int32, count)],
    )
    diffs = []
    for shift, output_index in ((0, 3), (15, 7)):
        expected = np.stack(
            [((xy[:, 0] + shift) % 30) - 15, ((xy[:, 1] + shift) % 30) - 15],
            axis=1,
        ).astype(np.float32)
        diffs.append(max_abs(outputs[output_index][: len(xy)], expected))
    np.testing.assert_array_equal(outputs[8].cpu().numpy(), [1, 1])
    diff = max(diffs)
    if diff != 0.0:
        raise AssertionError(f"DSVTRotatedSet position max_abs_diff={diff}")
    return diff


def check_dense_scatter(trt, logger, registry) -> float:
    features = np.zeros((MAX_PILLARS, 4), dtype=np.float32)
    features[:3] = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float32)
    coords = np.zeros((COORD_CAPACITY, 4), dtype=np.int32)
    coords[:3, 2:4] = np.asarray([[3, 2], [5, 4], [7, 6]], dtype=np.int32)
    count = np.asarray([3], dtype=np.int32)
    output = build_and_run(
        trt,
        logger,
        registry,
        "DSVTDenseScatter",
        [
            ("features", trt.float32, features),
            ("coords", trt.int32, coords),
            ("pillar_count", trt.int32, count),
        ],
    )[0]
    expected = np.zeros((1, 4, 360, 360), dtype=np.float32)
    for pillar in range(3):
        y, x = coords[pillar, 2:4]
        expected[0, :, y, x] = features[pillar]
    diff = max_abs(output, expected)
    if diff != 0.0:
        raise AssertionError(f"DSVTDenseScatter max_abs_diff={diff}")
    return diff


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    plugin_dir = args.plugin_dir.expanduser().resolve()
    for filename in PLUGIN_FILES:
        path = plugin_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)

    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    if not trt.init_libnvinfer_plugins(logger, ""):
        raise RuntimeError("failed to initialize TensorRT standard plugins")
    registry = trt.get_plugin_registry()
    checks = (
        ("DynamicPillarDecorate", check_decorate),
        ("DynamicScatterMax", check_scatter_max),
        ("TensorBarrier", check_barrier),
        ("DSVTRotatedSet", check_rotated_set),
        ("DSVTDenseScatter", check_dense_scatter),
    )
    for name, check in checks:
        diff = check(trt, logger, registry)
        print(f"PASS {name}: build=ok execute=ok finite=ok max_abs_diff={diff:.8g}")
    print(f"PASS all plugins TensorRT={trt.__version__}")


if __name__ == "__main__":
    main()
