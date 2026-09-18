"""TensorRT 8.5/10 engine builder with manifest and profile validation."""

import ctypes
import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


PLUGIN_FILES = (
    "libdynamic_pillar_decorate.so",
    "libdynamic_scatter_max.so",
    "libtensor_barrier.so",
    "libdsvt_rotated_set.so",
    "libdsvt_dense_scatter.so",
)


@dataclass(frozen=True)
class BuildOptions:
    precision: str = "fp16"
    workspace_gib: float = 4.0
    allow_tf32: bool = False
    optimization_level: int = 3
    max_aux_streams: int = 0
    timing_cache: Optional[Path] = None
    plugins: Tuple[Path, ...] = ()
    allow_random_init: bool = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def tensorrt_version(trt) -> Tuple[int, int]:
    try:
        parts = str(trt.__version__).split(".")
        return int(parts[0]), int(parts[1])
    except (AttributeError, IndexError, TypeError, ValueError) as error:
        raise RuntimeError("unable to determine the TensorRT version") from error


def tensorrt_major(trt) -> int:
    return tensorrt_version(trt)[0]


def configure_versioned_builder_options(trt, config, options: BuildOptions) -> None:
    major = tensorrt_major(trt)
    if major >= 10:
        config.builder_optimization_level = options.optimization_level
        config.max_aux_streams = options.max_aux_streams
        return
    if major == 8:
        if options.optimization_level != 3:
            raise ValueError(
                "TensorRT 8.5 does not expose builder_optimization_level; "
                "use the default value 3"
            )
        if options.max_aux_streams != 0:
            raise ValueError(
                "TensorRT 8.5 does not expose max_aux_streams; use 0"
            )
        return
    raise RuntimeError(f"unsupported TensorRT major version: {major}")


def load_json(path: Path, label: str) -> Dict[str, Any]:
    resolved = require_file(path, label)
    payload = json.loads(resolved.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a JSON object: {resolved}")
    return payload


def load_profile(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    return None if path is None else load_json(path, "optimization profile")


def _network_io_names(network) -> Tuple[Sequence[str], Sequence[str]]:
    inputs = [network.get_input(index).name for index in range(network.num_inputs)]
    outputs = [network.get_output(index).name for index in range(network.num_outputs)]
    return inputs, outputs


def _dtype_from_manifest(trt, dtype: str):
    names = {
        "float32": "FLOAT",
        "float16": "HALF",
        "int8": "INT8",
        "int32": "INT32",
        "int64": "INT64",
        "bool": "BOOL",
    }
    enum_name = names.get(dtype)
    if enum_name is None or not hasattr(trt.DataType, enum_name):
        raise ValueError(f"unsupported TensorRT manifest dtype: {dtype}")
    return getattr(trt.DataType, enum_name)


def validate_manifest_abi(network, trt, manifest: Mapping[str, Any], engine_name: str) -> None:
    if manifest.get("engine") != engine_name:
        raise ValueError(
            f"export manifest engine mismatch: expected={engine_name}, "
            f"actual={manifest.get('engine')}"
        )
    expected_inputs = manifest.get("inputs")
    expected_outputs = manifest.get("outputs")
    if not isinstance(expected_inputs, dict) or not isinstance(expected_outputs, dict):
        raise ValueError("export manifest is missing inputs/outputs objects")

    actual_inputs, actual_outputs = _network_io_names(network)
    if set(actual_inputs) != set(expected_inputs):
        raise ValueError(
            f"input ABI mismatch: manifest={list(expected_inputs)}, "
            f"onnx={actual_inputs}"
        )
    if set(actual_outputs) != set(expected_outputs):
        raise ValueError(
            f"output ABI mismatch: manifest={list(expected_outputs)}, "
            f"onnx={actual_outputs}"
        )

    tensors = {
        network.get_input(index).name: network.get_input(index)
        for index in range(network.num_inputs)
    }
    tensors.update({
        network.get_output(index).name: network.get_output(index)
        for index in range(network.num_outputs)
    })
    for name, spec in {**expected_inputs, **expected_outputs}.items():
        expected_dtype = _dtype_from_manifest(trt, spec["dtype"])
        if tensors[name].dtype != expected_dtype:
            raise ValueError(
                f"dtype mismatch for {name}: manifest={spec['dtype']}, "
                f"onnx={tensors[name].dtype}"
            )


def _shape_triplet(name: str, value: Any, network_shape: Sequence[int]):
    if not isinstance(value, dict) or set(value) != {"min", "opt", "max"}:
        raise ValueError(f"profile {name} must contain exactly min/opt/max")
    shapes = []
    for phase in ("min", "opt", "max"):
        shape = value[phase]
        if not isinstance(shape, list) or len(shape) != len(network_shape):
            raise ValueError(
                f"profile {name}.{phase} rank must be {len(network_shape)}"
            )
        if any(isinstance(axis, bool) or not isinstance(axis, int) or axis <= 0 for axis in shape):
            raise ValueError(f"profile {name}.{phase} axes must be positive integers")
        for axis, (declared, actual) in enumerate(zip(shape, network_shape)):
            if actual >= 0 and declared != actual:
                raise ValueError(
                    f"profile {name}.{phase}[{axis}]={declared} "
                    f"does not match ONNX static axis {actual}"
                )
        shapes.append(tuple(shape))
    for axis, values in enumerate(zip(*shapes)):
        if not values[0] <= values[1] <= values[2]:
            raise ValueError(
                f"profile {name} axis {axis} must satisfy min <= opt <= max: "
                f"{values}"
            )
    return tuple(shapes)


def validate_dsvt_profile(profile: Mapping[str, Any]) -> None:
    def axis(name: str, phase: str, index: int) -> int:
        return int(profile[name][phase][index])

    for phase in ("min", "opt", "max"):
        pillar_counts = {
            axis("src", phase, 0),
            axis("gather_shift_0", phase, 1),
            axis("gather_shift_1", phase, 1),
            axis("position_embeddings", phase, 2),
            axis("coords", phase, 0),
        }
        if len(pillar_counts) != 1:
            raise ValueError(
                f"DSVT {phase} profile has inconsistent pillar capacities: "
                f"{sorted(pillar_counts)}"
            )
        if axis("set_indices_shift_0", phase, 1) != axis("set_masks_shift_0", phase, 1):
            raise ValueError(f"DSVT {phase} shift-0 index/mask set counts differ")
        if axis("set_indices_shift_1", phase, 1) != axis("set_masks_shift_1", phase, 1):
            raise ValueError(f"DSVT {phase} shift-1 index/mask set counts differ")


def add_optimization_profile(builder, config, network, profile_data, engine_name):
    dynamic_inputs = {}
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        shape = tuple(int(axis) for axis in tensor.shape)
        if any(axis < 0 for axis in shape):
            dynamic_inputs[tensor.name] = shape

    if not dynamic_inputs:
        if profile_data:
            raise ValueError(
                f"{engine_name} has static inputs and must not receive a profile"
            )
        return None
    if profile_data is None:
        raise ValueError(
            f"{engine_name} has dynamic inputs {sorted(dynamic_inputs)}; "
            "--profile is required"
        )
    if set(profile_data) != set(dynamic_inputs):
        raise ValueError(
            f"profile input names differ: expected={sorted(dynamic_inputs)}, "
            f"actual={sorted(profile_data)}"
        )

    parsed = {
        name: _shape_triplet(name, profile_data[name], network_shape)
        for name, network_shape in dynamic_inputs.items()
    }
    if engine_name == "dsvt_bev":
        validate_dsvt_profile(profile_data)

    profile = builder.create_optimization_profile()
    for name, (minimum, optimum, maximum) in parsed.items():
        accepted = profile.set_shape(name, minimum, optimum, maximum)
        if accepted is False:
            raise RuntimeError(f"TensorRT rejected optimization profile for {name}")
    if config.add_optimization_profile(profile) < 0:
        raise RuntimeError("TensorRT rejected the optimization profile")
    return parsed


def _binding_specs(engine, trt, profile_data):
    specs = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        mode = engine.get_tensor_mode(name)
        spec = {
            "name": name,
            "mode": "input" if mode == trt.TensorIOMode.INPUT else "output",
            "dtype": str(engine.get_tensor_dtype(name)),
            "shape": [int(axis) for axis in engine.get_tensor_shape(name)],
            "location": str(engine.get_tensor_location(name)),
        }
        if name in (profile_data or {}):
            spec["profile"] = profile_data[name]
        specs.append(spec)
    return specs


def build_engine(
    engine_name: str,
    onnx_path: Path,
    export_manifest_path: Path,
    engine_path: Path,
    build_manifest_path: Path,
    options: BuildOptions,
    profile_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if options.precision not in {"fp16", "fp32"}:
        raise ValueError("precision must be fp16 or fp32")
    if options.workspace_gib <= 0:
        raise ValueError("workspace_gib must be positive")
    if not 0 <= options.optimization_level <= 5:
        raise ValueError("optimization_level must be in [0, 5]")
    if options.max_aux_streams < 0:
        raise ValueError("max_aux_streams must be non-negative")

    import tensorrt as trt

    onnx_path = require_file(onnx_path, "ONNX")
    export_manifest_path = require_file(export_manifest_path, "export manifest")
    export_manifest = load_json(export_manifest_path, "export manifest")
    model = export_manifest.get("model", {})
    if model.get("random_init_diagnostic") and not options.allow_random_init:
        raise ValueError(
            "refusing to build a random-init ONNX; pass --allow-random-init "
            "only for a diagnostic engine"
        )

    plugin_paths = tuple(require_file(path, "plugin") for path in options.plugins)
    for path in plugin_paths:
        ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)

    logger = trt.Logger(trt.Logger.INFO)
    if not trt.init_libnvinfer_plugins(logger, ""):
        raise RuntimeError("failed to initialize TensorRT standard plugins")
    builder = trt.Builder(logger)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError("TensorRT ONNX parse failed:\n" + "\n".join(errors))

    validate_manifest_abi(network, trt, export_manifest, engine_name)
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

    parsed_profile = add_optimization_profile(
        builder, config, network, profile_data, engine_name
    )

    timing_cache_path = None
    if options.timing_cache is not None:
        timing_cache_path = Path(options.timing_cache).expanduser().resolve()
        cache_data = timing_cache_path.read_bytes() if timing_cache_path.is_file() else b""
        timing_cache = config.create_timing_cache(cache_data)
        if not config.set_timing_cache(timing_cache, False):
            raise RuntimeError(f"TensorRT rejected timing cache {timing_cache_path}")

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT failed to build {engine_name}")

    engine_path = Path(engine_path).expanduser().resolve()
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_engine = engine_path.with_suffix(engine_path.suffix + ".tmp")
    temporary_engine.write_bytes(bytes(serialized))
    temporary_engine.replace(engine_path)

    if timing_cache_path is not None:
        timing_cache_path.parent.mkdir(parents=True, exist_ok=True)
        timing_cache_path.write_bytes(bytes(config.get_timing_cache().serialize()))

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError("built engine failed deserialize-readback")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("built engine failed execution-context creation")
    bindings = _binding_specs(engine, trt, profile_data)
    device_memory = getattr(engine, "device_memory_size_v2", None)
    if device_memory is None:
        device_memory = engine.device_memory_size

    manifest = {
        "schema_version": 1,
        "engine_name": engine_name,
        "engine": str(engine_path),
        "engine_sha256": sha256_file(engine_path),
        "onnx": str(onnx_path),
        "onnx_sha256": sha256_file(onnx_path),
        "export_manifest": str(export_manifest_path),
        "export_manifest_sha256": sha256_file(export_manifest_path),
        "model": model,
        "precision": options.precision,
        "tf32": options.allow_tf32,
        "workspace_gib": options.workspace_gib,
        "builder_optimization_level": options.optimization_level,
        "max_aux_streams": options.max_aux_streams,
        "timing_cache": str(timing_cache_path) if timing_cache_path else None,
        "tensorrt_version": trt.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device_memory_size": int(device_memory),
        "bindings": bindings,
        "optimization_profile": profile_data,
        "plugins": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in plugin_paths
        ],
    }
    build_manifest_path = Path(build_manifest_path).expanduser().resolve()
    build_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    build_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest
