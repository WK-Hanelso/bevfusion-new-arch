"""Shared device contract for deployment exporters."""

from typing import Any, Dict

import torch


CUDA_REQUIRED_MESSAGE = "CUDA is required for this deployment export"


def require_export_device(device: str, allow_cpu_only: bool) -> Dict[str, Any]:
    """Validate the export device and return its manifest provenance."""

    if not torch.cuda.is_available():
        if not allow_cpu_only:
            raise RuntimeError(CUDA_REQUIRED_MESSAGE)
        try:
            requested_device = torch.device(device)
        except (RuntimeError, TypeError, ValueError):
            raise RuntimeError(CUDA_REQUIRED_MESSAGE) from None
        if requested_device.type != "cpu":
            raise RuntimeError(CUDA_REQUIRED_MESSAGE)
    else:
        requested_device = torch.device(device)

    cpu_only = allow_cpu_only and requested_device.type == "cpu"
    return {
        "export_device": str(requested_device),
        "cpu_only": cpu_only,
    }
