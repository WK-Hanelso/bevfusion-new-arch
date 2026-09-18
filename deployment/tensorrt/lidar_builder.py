"""Compose the raw-point DSVT TensorRT engine from exported learned artifacts."""

import ctypes
import json
import platform
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import onnx
from onnx import compose

from deployment.tensorrt.builder import (
    BuildOptions,
    PLUGIN_FILES,
    configure_versioned_builder_options,
    require_file,
    sha256_file,
    tensorrt_major,
)
BACKBONE_INPUTS = (
    "src",
    "set_indices_shift_0",
    "set_indices_shift_1",
    "set_masks_shift_0",
    "set_masks_shift_1",
    "gather_shift_0",
    "gather_shift_1",
    "position_embeddings",
)


def load_lidar_artifacts(directory: Path, allow_random_init: bool):
    directory = Path(directory).expanduser().resolve()
    manifest_path = require_file(directory / "lidar_trt.manifest.json", "LiDAR manifest")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("engine") != "lidar_raw":
        raise ValueError(f"invalid LiDAR artifact type: {manifest.get('engine')}")
    model = manifest.get("model")
    if not isinstance(model, dict):
        raise ValueError("LiDAR manifest is missing model provenance")
    if model.get("random_init_diagnostic") and not allow_random_init:
        raise ValueError("refusing random-init LiDAR artifacts for production build")
    if type(manifest.get("official_layout")) is not bool:
        raise ValueError("LiDAR manifest is missing a boolean official_layout")
    zero_feature_channels = manifest.get("zero_feature_channels")
    if (
        not isinstance(zero_feature_channels, list)
        or any(
            type(channel) is not int or not 0 <= channel < 5
            for channel in zero_feature_channels
        )
        or len(set(zero_feature_channels)) != len(zero_feature_channels)
    ):
        raise ValueError("LiDAR manifest has invalid zero_feature_channels")

    resolved = {}
    artifacts = manifest.get("artifacts", {})
    for name in ("backbone", "neck", "frontend_weights"):
        spec = artifacts.get(name)
        if not isinstance(spec, dict) or "file" not in spec or "sha256" not in spec:
            raise ValueError(f"LiDAR manifest is missing artifact {name}")
        path = require_file(directory / spec["file"], f"LiDAR {name}")
        actual_hash = sha256_file(path)
        if actual_hash != spec["sha256"]:
            raise ValueError(
                f"LiDAR {name} SHA-256 mismatch: expected={spec['sha256']}, "
                f"actual={actual_hash}"
            )
        resolved[name] = path
    return manifest_path, manifest, resolved


def load_frontend_weights(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        weights = {
            name: np.ascontiguousarray(archive[name].astype(np.float32, copy=False))
            for name in archive.files
        }
    required = set()
    for index in range(2):
        required.update(
            f"pfn{index}.{suffix}" for suffix in ("weight", "scale", "shift")
        )
    for block in range(4):
        for shift in range(2):
            prefix = f"position.{block}.{shift}"
            required.update(
                prefix + suffix
                for suffix in (
                    ".first_weight",
                    ".first_scale",
                    ".first_shift",
                    ".second_weight",
                    ".second_bias",
                )
            )
    missing = required.difference(weights)
    if missing:
        raise ValueError(f"frontend weights are missing arrays: {sorted(missing)}")
    return weights


def parse_onnx(parser, payload: bytes) -> None:
    if parser.parse(payload):
        return
    errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
    raise RuntimeError("TensorRT ONNX parse failed:\n" + "\n".join(errors))


def network_input(network, name):
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        if tensor.name == name:
            return tensor
    raise KeyError(f"network input not found: {name}")


def network_output(network, name):
    for index in range(network.num_outputs):
        tensor = network.get_output(index)
        if tensor.name == name:
            return tensor
    raise KeyError(f"network output not found: {name}")


def replace_tensor(network, old, new) -> None:
    consumers = 0
    for layer_index in range(network.num_layers):
        layer = network.get_layer(layer_index)
        for input_index in range(layer.num_inputs):
            current = layer.get_input(input_index)
            if current is not None and current.name == old.name:
                layer.set_input(input_index, new)
                consumers += 1
    if consumers == 0:
        raise RuntimeError(f"no consumers found for TensorRT tensor {old.name}")
    network.remove_tensor(old)


def _get_plugin_creator(registry, name, version):
    if hasattr(registry, "get_creator"):
        return registry.get_creator(name, version, "")
    return registry.get_plugin_creator(name, version, "")


def create_plugin(trt, registry, name, instance):
    creator = _get_plugin_creator(registry, name, "1")
    if creator is None:
        raise RuntimeError(f"TensorRT plugin creator missing: {name}:1")
    fields = trt.PluginFieldCollection([])
    if tensorrt_major(trt) >= 10:
        plugin = creator.create_plugin(instance, fields, trt.TensorRTPhase.BUILD)
    else:
        plugin = creator.create_plugin(instance, fields)
    if plugin is None:
        raise RuntimeError(f"TensorRT plugin creation failed: {name}:1")
    return plugin


def add_plugin(trt, network, inputs, plugin):
    if tensorrt_major(trt) >= 10:
        return network.add_plugin_v3(inputs, [], plugin)
    return network.add_plugin_v2(inputs, plugin)


def add_linear_bn_relu(trt, network, tensor, weights, prefix, weight_store):
    weight = weights[prefix + ".weight"]
    scale = weights[prefix + ".scale"]
    shift = weights[prefix + ".shift"]
    weight_store.extend((weight, scale, shift))
    weight_layer = network.add_constant(weight.shape, weight)
    projected = network.add_matrix_multiply(
        tensor,
        trt.MatrixOperation.NONE,
        weight_layer.get_output(0),
        trt.MatrixOperation.NONE,
    )
    scale_layer = network.add_constant((1, scale.size), scale.reshape(1, -1))
    shift_layer = network.add_constant((1, shift.size), shift.reshape(1, -1))
    scaled = network.add_elementwise(
        projected.get_output(0),
        scale_layer.get_output(0),
        trt.ElementWiseOperation.PROD,
    )
    shifted = network.add_elementwise(
        scaled.get_output(0),
        shift_layer.get_output(0),
        trt.ElementWiseOperation.SUM,
    )
    return network.add_activation(
        shifted.get_output(0), trt.ActivationType.RELU
    ).get_output(0)


def add_position_embedding(trt, network, coordinates, weights, prefix, weight_store):
    first_weight = weights[prefix + ".first_weight"]
    first_scale = weights[prefix + ".first_scale"]
    first_shift = weights[prefix + ".first_shift"]
    second_weight = weights[prefix + ".second_weight"]
    second_bias = weights[prefix + ".second_bias"]
    weight_store.extend(
        (first_weight, first_scale, first_shift, second_weight, second_bias)
    )

    first_layer = network.add_constant(first_weight.shape, first_weight)
    projected = network.add_matrix_multiply(
        coordinates,
        trt.MatrixOperation.NONE,
        first_layer.get_output(0),
        trt.MatrixOperation.NONE,
    )
    scale_layer = network.add_constant(
        (1, first_scale.size), first_scale.reshape(1, -1)
    )
    shift_layer = network.add_constant(
        (1, first_shift.size), first_shift.reshape(1, -1)
    )
    scaled = network.add_elementwise(
        projected.get_output(0),
        scale_layer.get_output(0),
        trt.ElementWiseOperation.PROD,
    )
    shifted = network.add_elementwise(
        scaled.get_output(0),
        shift_layer.get_output(0),
        trt.ElementWiseOperation.SUM,
    )
    hidden = network.add_activation(
        shifted.get_output(0), trt.ActivationType.RELU
    )
    second_layer = network.add_constant(second_weight.shape, second_weight)
    projected = network.add_matrix_multiply(
        hidden.get_output(0),
        trt.MatrixOperation.NONE,
        second_layer.get_output(0),
        trt.MatrixOperation.NONE,
    )
    bias_layer = network.add_constant(
        (1, second_bias.size), second_bias.reshape(1, -1)
    )
    return network.add_elementwise(
        projected.get_output(0),
        bias_layer.get_output(0),
        trt.ElementWiseOperation.SUM,
    ).get_output(0)


def _expect_shape(tensor, expected: Sequence[int], name: str) -> None:
    actual = tuple(int(axis) for axis in tensor.shape)
    if actual != tuple(expected):
        raise ValueError(
            f"plugin/artifact capacity mismatch for {name}: "
            f"expected={tuple(expected)}, actual={actual}"
        )


def add_frontend(trt, builder, network, registry, weights, capacity, point_profile, weight_store):
    max_pillars = int(capacity["max_pillars"])
    max_sets = int(capacity["max_sets_per_shift"])
    coord_capacity = int(capacity["coordinate_capacity"])
    set_size = int(capacity["set_size"])

    points = network.add_input("points", trt.float32, (-1, 5))
    geometry = add_plugin(
        trt,
        network,
        [points],
        create_plugin(trt, registry, "DynamicPillarDecorate", "pillar_geometry"),
    )
    decorated, inverse, coords, pillar_count = (
        geometry.get_output(index) for index in range(4)
    )
    zero_i32 = np.asarray([0], dtype=np.int32)
    minus_one_i32 = np.asarray([-1], dtype=np.int32)
    weight_store.extend((zero_i32, minus_one_i32))
    zero = network.add_constant((1,), zero_i32).get_output(0)
    minus_one = network.add_constant((1,), minus_one_i32).get_output(0)
    safe_inverse = network.add_elementwise(
        inverse, zero, trt.ElementWiseOperation.MAX
    ).get_output(0)
    valid = network.add_elementwise(
        inverse, minus_one, trt.ElementWiseOperation.GREATER
    ).get_output(0)
    valid_float = network.add_cast(valid, trt.float32).get_output(0)
    valid_column = network.add_shuffle(valid_float)
    valid_column.reshape_dims = (-1, 1)

    first = add_linear_bn_relu(
        trt, network, decorated, weights, "pfn0", weight_store
    )
    first = add_plugin(
        trt,
        network,
        [first],
        create_plugin(trt, registry, "TensorBarrier", "pfn1_barrier"),
    ).get_output(0)
    first = network.add_elementwise(
        first, valid_column.get_output(0), trt.ElementWiseOperation.PROD
    ).get_output(0)
    pooled = add_plugin(
        trt,
        network,
        [first, safe_inverse],
        create_plugin(trt, registry, "DynamicScatterMax", "pfn1_scatter"),
    ).get_output(0)
    gathered = network.add_gather(pooled, safe_inverse, axis=0).get_output(0)
    concatenated = network.add_concatenation([first, gathered])
    concatenated.axis = 1
    second = add_linear_bn_relu(
        trt, network, concatenated.get_output(0), weights, "pfn1", weight_store
    )
    second = network.add_elementwise(
        second, valid_column.get_output(0), trt.ElementWiseOperation.PROD
    ).get_output(0)
    src = add_plugin(
        trt,
        network,
        [second, safe_inverse],
        create_plugin(trt, registry, "DynamicScatterMax", "pfn2_scatter"),
    ).get_output(0)

    rotated = add_plugin(
        trt,
        network,
        [coords, pillar_count],
        create_plugin(trt, registry, "DSVTRotatedSet", "rotated_sets"),
    )
    positions = []
    for block in range(4):
        for shift in range(2):
            position = add_position_embedding(
                trt,
                network,
                rotated.get_output(shift * 4 + 3),
                weights,
                f"position.{block}.{shift}",
                weight_store,
            )
            shaped = network.add_shuffle(position)
            shaped.reshape_dims = (1, 1, max_pillars, 128)
            positions.append(shaped.get_output(0))
    position_concat = network.add_concatenation(positions)
    position_concat.axis = 1
    position_reshape = network.add_shuffle(position_concat.get_output(0))
    position_reshape.reshape_dims = (4, 2, max_pillars, 128)

    frontend = {
        "src": src,
        "set_indices_shift_0": rotated.get_output(0),
        "set_indices_shift_1": rotated.get_output(4),
        "set_masks_shift_0": rotated.get_output(1),
        "set_masks_shift_1": rotated.get_output(5),
        "gather_shift_0": rotated.get_output(2),
        "gather_shift_1": rotated.get_output(6),
        "position_embeddings": position_reshape.get_output(0),
        "coords": coords,
        "pillar_count": pillar_count,
        "set_counts": rotated.get_output(8),
    }
    _expect_shape(src, (max_pillars, 128), "src")
    _expect_shape(coords, (coord_capacity, 4), "coords")
    for shift in range(2):
        _expect_shape(
            frontend[f"set_indices_shift_{shift}"],
            (2, max_sets, set_size),
            f"set_indices_shift_{shift}",
        )
        _expect_shape(
            frontend[f"set_masks_shift_{shift}"],
            (2, max_sets, set_size),
            f"set_masks_shift_{shift}",
        )
        _expect_shape(
            frontend[f"gather_shift_{shift}"],
            (2, max_pillars),
            f"gather_shift_{shift}",
        )
    _expect_shape(
        frontend["position_embeddings"],
        (4, 2, max_pillars, 128),
        "position_embeddings",
    )

    profile = builder.create_optimization_profile()
    if profile.set_shape(
        "points",
        (point_profile[0], 5),
        (point_profile[1], 5),
        (point_profile[2], 5),
    ) is False:
        raise RuntimeError("TensorRT rejected raw-point optimization profile")
    return profile, frontend


def add_capacity_status(trt, network, pillar_count, set_counts, max_pillars, weight_store):
    maximum = np.asarray([max_pillars], dtype=np.int32)
    zero = np.asarray([0], dtype=np.int32)
    weight_store.extend((maximum, zero))
    maximum_tensor = network.add_constant((1,), maximum).get_output(0)
    zero_tensor = network.add_constant((1,), zero).get_output(0)
    pillar_overflow = network.add_cast(
        network.add_elementwise(
            pillar_count, maximum_tensor, trt.ElementWiseOperation.GREATER
        ).get_output(0),
        trt.int32,
    ).get_output(0)
    set_overflow = network.add_cast(
        network.add_elementwise(
            set_counts, zero_tensor, trt.ElementWiseOperation.LESS
        ).get_output(0),
        trt.int32,
    ).get_output(0)
    any_set_overflow = network.add_reduce(
        set_overflow, trt.ReduceOperation.MAX, 1, True
    ).get_output(0)
    status = network.add_elementwise(
        pillar_overflow, any_set_overflow, trt.ElementWiseOperation.MAX
    ).get_output(0)
    status.name = "lidar_status"
    return status


def validate_point_profile(profile: Tuple[int, int, int], max_points: int) -> None:
    if any(isinstance(value, bool) or not isinstance(value, int) for value in profile):
        raise TypeError("point profile values must be integers")
    if not 1 <= profile[0] <= profile[1] <= profile[2] <= max_points:
        raise ValueError(
            "point profile must satisfy "
            f"1 <= min <= opt <= max <= {max_points}: {profile}"
        )


def build_raw_lidar_engine(
    artifact_dir: Path,
    plugin_dir: Path,
    engine_path: Path,
    build_manifest_path: Path,
    point_profile: Tuple[int, int, int],
    options: BuildOptions,
) -> Dict[str, Any]:
    if options.precision not in {"fp16", "fp32"}:
        raise ValueError("precision must be fp16 or fp32")
    if options.workspace_gib <= 0:
        raise ValueError("workspace_gib must be positive")
    if not 0 <= options.optimization_level <= 5:
        raise ValueError("optimization_level must be in [0,5]")
    if options.max_aux_streams < 0:
        raise ValueError("max_aux_streams must be non-negative")

    import tensorrt as trt

    manifest_path, artifact_manifest, artifacts = load_lidar_artifacts(
        artifact_dir, options.allow_random_init
    )
    capacity = artifact_manifest.get("capacity", {})
    max_points = int(capacity.get("max_points", 0))
    validate_point_profile(point_profile, max_points)
    weights = load_frontend_weights(artifacts["frontend_weights"])

    plugin_dir = Path(plugin_dir).expanduser().resolve()
    plugin_paths = tuple(
        require_file(plugin_dir / name, "LiDAR plugin") for name in PLUGIN_FILES
    )
    for path in plugin_paths:
        ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)

    logger = trt.Logger(trt.Logger.INFO)
    if not trt.init_libnvinfer_plugins(logger, ""):
        raise RuntimeError("failed to initialize TensorRT standard plugins")
    builder = trt.Builder(logger)
    if tensorrt_major(trt) >= 10:
        network_flags = 0
    else:
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    backbone_model = onnx.load(str(artifacts["backbone"]), load_external_data=False)
    neck_model = onnx.load(str(artifacts["neck"]), load_external_data=False)
    merged = compose.merge_models(
        backbone_model, neck_model, io_map=[], prefix2="neck__"
    )
    onnx.checker.check_model(merged)
    parse_onnx(parser, merged.SerializeToString())

    old_inputs = {name: network_input(network, name) for name in BACKBONE_INPUTS}
    transformed = network_output(network, "transformed_pillars")
    network.unmark_output(transformed)
    old_dense = network_input(network, "neck__dense_bev")
    lidar_bev = network_output(network, "neck__lidar_bev")
    lidar_bev.name = "lidar_bev"

    registry = trt.get_plugin_registry()
    weight_store = []
    profile, frontend = add_frontend(
        trt,
        builder,
        network,
        registry,
        weights,
        capacity,
        point_profile,
        weight_store,
    )
    for name, old in old_inputs.items():
        replace_tensor(network, old, frontend[name])
    dense = add_plugin(
        trt,
        network,
        [transformed, frontend["coords"], frontend["pillar_count"]],
        create_plugin(trt, registry, "DSVTDenseScatter", "dense_scatter"),
    ).get_output(0)
    replace_tensor(network, old_dense, dense)
    lidar_status = add_capacity_status(
        trt,
        network,
        frontend["pillar_count"],
        frontend["set_counts"],
        int(capacity["max_pillars"]),
        weight_store,
    )
    network.mark_output(lidar_status)

    public_inputs = [network.get_input(i).name for i in range(network.num_inputs)]
    public_outputs = [network.get_output(i).name for i in range(network.num_outputs)]
    if public_inputs != ["points"] or set(public_outputs) != {"lidar_bev", "lidar_status"}:
        raise RuntimeError(
            f"invalid raw LiDAR ABI: inputs={public_inputs}, outputs={public_outputs}"
        )

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(options.workspace_gib * (1 << 30))
    )
    configure_versioned_builder_options(trt, config, options)
    if options.allow_tf32:
        config.set_flag(trt.BuilderFlag.TF32)
    else:
        config.clear_flag(trt.BuilderFlag.TF32)
    if options.precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    if config.add_optimization_profile(profile) < 0:
        raise RuntimeError("TensorRT rejected raw-point optimization profile")

    timing_cache_path = None
    if options.timing_cache is not None:
        timing_cache_path = Path(options.timing_cache).expanduser().resolve()
        cache_data = timing_cache_path.read_bytes() if timing_cache_path.is_file() else b""
        timing_cache = config.create_timing_cache(cache_data)
        if not config.set_timing_cache(timing_cache, False):
            raise RuntimeError(f"TensorRT rejected timing cache {timing_cache_path}")

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build raw-point LiDAR engine")
    engine_path = Path(engine_path).expanduser().resolve()
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = engine_path.with_suffix(engine_path.suffix + ".tmp")
    temporary.write_bytes(bytes(serialized))
    temporary.replace(engine_path)
    if timing_cache_path is not None:
        timing_cache_path.parent.mkdir(parents=True, exist_ok=True)
        timing_cache_path.write_bytes(bytes(config.get_timing_cache().serialize()))

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError("raw-point LiDAR engine deserialize-readback failed")
    context = engine.create_execution_context()
    if context is None or not context.set_input_shape(
        "points", (point_profile[1], 5)
    ):
        raise RuntimeError("raw-point LiDAR execution context/profile check failed")
    output_shapes = {
        name: [int(axis) for axis in context.get_tensor_shape(name)]
        for name in ("lidar_bev", "lidar_status")
    }
    if output_shapes != {"lidar_bev": [1, 256, 180, 180], "lidar_status": [1]}:
        raise RuntimeError(f"raw-point LiDAR output ABI mismatch: {output_shapes}")

    device_memory = getattr(engine, "device_memory_size_v2", None)
    if device_memory is None:
        device_memory = engine.device_memory_size
    build_manifest = {
        "schema_version": 1,
        "engine_name": "lidar_raw",
        "engine": str(engine_path),
        "engine_sha256": sha256_file(engine_path),
        "artifact_manifest": str(manifest_path),
        "artifact_manifest_sha256": sha256_file(manifest_path),
        "model": artifact_manifest["model"],
        "official_layout": artifact_manifest["official_layout"],
        "zero_feature_channels": artifact_manifest["zero_feature_channels"],
        "precision": options.precision,
        "tf32": options.allow_tf32,
        "workspace_gib": options.workspace_gib,
        "builder_optimization_level": options.optimization_level,
        "max_aux_streams": options.max_aux_streams,
        "point_profile": {
            "min": [point_profile[0], 5],
            "opt": [point_profile[1], 5],
            "max": [point_profile[2], 5],
        },
        "capacity": capacity,
        "bindings": {
            "points": {"dtype": "float32", "shape": [-1, 5]},
            "lidar_bev": {
                "dtype": str(engine.get_tensor_dtype("lidar_bev")),
                "shape": output_shapes["lidar_bev"],
            },
            "lidar_status": {
                "dtype": str(engine.get_tensor_dtype("lidar_status")),
                "shape": output_shapes["lidar_status"],
                "semantics": "0=valid, nonzero=capacity overflow",
            },
        },
        "plugins": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in plugin_paths
        ],
        "timing_cache": str(timing_cache_path) if timing_cache_path else None,
        "tensorrt_version": trt.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device_memory_size": int(device_memory),
    }
    build_manifest_path = Path(build_manifest_path).expanduser().resolve()
    build_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    build_manifest_path.write_text(
        json.dumps(build_manifest, indent=2, sort_keys=True) + "\n"
    )
    return build_manifest
