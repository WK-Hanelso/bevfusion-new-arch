"""Compatibility helpers for the bundled spconv v1 fork and spconv v2.

This module intentionally has no mmcv or mmdet3d imports at module import time.
That keeps the state-dict conversion helpers usable in lightweight CPU-only
environments.
"""

import importlib
import os
from collections import OrderedDict


SPCONV_ENV = "BEVFUSION_SPCONV"
_BACKENDS = ("legacy", "v2")


def resolve_spconv_backend(backend=None):
    """Resolve a config override or ``BEVFUSION_SPCONV``.

    The legacy backend remains the default.  A config value takes precedence
    over the environment so two encoders can be instantiated side-by-side in
    equivalence tests.
    """

    selected = backend if backend is not None else os.getenv(SPCONV_ENV, "legacy")
    selected = str(selected).strip().lower()
    if selected not in _BACKENDS:
        raise ValueError(
            f"Unsupported spconv backend {selected!r}; expected one of {_BACKENDS}"
        )
    return selected


def get_spconv_module(backend=None):
    """Return the selected sparse convolution module, importing it lazily."""

    selected = resolve_spconv_backend(backend)
    if selected == "legacy":
        return importlib.import_module("mmdet3d.ops.spconv")
    try:
        return importlib.import_module("spconv.pytorch")
    except ImportError as exc:
        raise ImportError(
            "spconv v2 was selected, but spconv.pytorch is unavailable. "
            "Install a compatible spconv-cu12x wheel or select the legacy backend."
        ) from exc


def legacy_to_spconv2_weight(weight):
    """Convert a v1 ``(*kernel, in, out)`` weight to v2 layout.

    spconv v2 uses ``(out, *kernel, in)``.  The implementation is dimension
    independent, although BEVFusion currently uses 3-D convolutions.
    """

    if weight.ndim < 3:
        raise ValueError("a sparse convolution weight must have at least 3 dimensions")
    return weight.permute(
        weight.ndim - 1, *range(weight.ndim - 2), weight.ndim - 2
    ).contiguous()


def spconv2_to_legacy_weight(weight):
    """Convert a v2 ``(out, *kernel, in)`` weight to v1 layout."""

    if weight.ndim < 3:
        raise ValueError("a sparse convolution weight must have at least 3 dimensions")
    return weight.permute(
        *range(1, weight.ndim - 1), weight.ndim - 1, 0
    ).contiguous()


def map_spconv_state_dict_key(key, target_keys):
    """Map a legacy key to the matching v2 key.

    Both backends are built with the same module hierarchy, so this mapping is
    deliberately identity-only.  Keeping it explicit makes key compatibility
    testable and prevents silently guessing unrelated checkpoint prefixes.
    """

    return key if key in target_keys else None


def convert_spconv_state_dict(
    state_dict, target_state_dict, source_backend="legacy", target_backend="v2"
):
    """Return a state dict converted between sparse convolution backends.

    Non-convolution tensors and unknown keys are preserved so that PyTorch's
    normal ``strict`` handling remains authoritative.  A tensor is permuted
    only when the converted shape agrees with the target tensor.
    """

    source_backend = resolve_spconv_backend(source_backend)
    target_backend = resolve_spconv_backend(target_backend)
    target_keys = set(target_state_dict)
    converted = OrderedDict()

    for key, value in state_dict.items():
        target_key = map_spconv_state_dict_key(key, target_keys)
        output = value
        if target_key is not None and getattr(value, "ndim", 0) >= 3:
            target = target_state_dict[target_key]
            if source_backend == "legacy" and target_backend == "v2":
                candidate = legacy_to_spconv2_weight(value)
            elif source_backend == "v2" and target_backend == "legacy":
                candidate = spconv2_to_legacy_weight(value)
            else:
                candidate = value
            if candidate.shape == target.shape:
                output = candidate
        converted[key] = output

    if hasattr(state_dict, "_metadata"):
        converted._metadata = state_dict._metadata
    return converted


def register_legacy_weight_load_hook(module):
    """Let a v2 convolution accept a legacy-shaped weight during recursion.

    This covers loading a full detector checkpoint, where calling a custom
    ``SparseEncoder.load_state_dict`` method would otherwise be bypassed.
    Already-v2 weights are left untouched.
    """

    def convert_weight(state_dict, prefix, *args):
        key = prefix + "weight"
        if key not in state_dict:
            return
        value = state_dict[key]
        target = module.weight
        if value.shape == target.shape or getattr(value, "ndim", 0) < 3:
            return
        candidate = legacy_to_spconv2_weight(value)
        if candidate.shape == target.shape:
            state_dict[key] = candidate

    module._register_load_state_dict_pre_hook(convert_weight)
    return module
