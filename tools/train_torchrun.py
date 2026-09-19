"""Run ``tools/train.py`` under ``torchrun`` instead of ``torchpack dist-run``.

``torchpack.distributed.init`` requires ``mpi4py`` and a ``MASTER_HOST``
environment variable.  Neither is available on this workstation, so this shim
fills the same torchpack process-group state from the standard torchrun
environment variables and then executes the unmodified training entry point.

Usage:

    torchrun --nproc_per_node=1 tools/train_torchrun.py <config> --run-dir <dir>
"""

import distutils.version  # noqa: F401  torch.utils.tensorboard reads this lazily
import os
import runpy
import sys

import torch
import torch.distributed

from torchpack import distributed as dist
from torchpack.distributed import context


def _init_from_torchrun_env(backend: str = "nccl", **_) -> None:
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    world_rank = int(os.environ.get("RANK", 0))
    local_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend=backend,
            init_method="env://",
            world_size=world_size,
            rank=world_rank,
        )

    context._world_size, context._world_rank = world_size, world_rank
    context._local_size, context._local_rank = local_size, local_rank


def _patch_yapf() -> None:
    """mmcv 1.4.0 calls ``yapf.FormatCode(..., verify=True)``.

    The installed yapf dropped that keyword, which breaks ``Config.pretty_text``
    before training starts.  Accept and ignore it.
    """

    import yapf.yapflib.yapf_api as yapf_api

    original = yapf_api.FormatCode
    if getattr(original, "_verify_patched", False):
        return

    def format_code(*args, **kwargs):
        kwargs.pop("verify", None)
        return original(*args, **kwargs)

    format_code._verify_patched = True
    yapf_api.FormatCode = format_code

    import mmcv.utils.config as mmcv_config

    mmcv_config.FormatCode = format_code


def _patch_mmcv_get_stream() -> None:
    """mmcv 1.x passes an int GPU id to ``torch.nn.parallel._functions._get_stream``.

    torch >= 2.1 expects a ``torch.device`` there (``device.type``), which breaks
    ``MMDistributedDataParallel.scatter`` with ``'int' object has no attribute
    'type'``.  Wrap ints into ``torch.device`` for mmcv's copy of the symbol.
    """

    try:
        import mmcv.parallel._functions as mmcv_functions
        from torch.nn.parallel import _functions as torch_functions
    except ImportError:
        return

    original = torch_functions._get_stream
    if getattr(mmcv_functions._get_stream, "_device_patched", False):
        return

    def get_stream(device):
        if isinstance(device, int):
            device = torch.device("cuda", device)
        return original(device)

    get_stream._device_patched = True
    mmcv_functions._get_stream = get_stream


def _patch_mmcv_ddp_forward() -> None:
    """mmcv 1.7 ``MMDistributedDataParallel._run_ddp_forward`` reads
    ``self._use_replicated_tensor_module``, which torch >= 2.1 removed.  Only the
    evaluation path (``model(...)`` -> ``DDP.forward``) hits it; ``train_step``
    does not.  Re-implement it without the removed attribute, keeping mmcv's
    ``to_kwargs`` so DataContainer inputs are still scattered.
    """

    try:
        from mmcv.parallel.distributed import MMDistributedDataParallel
    except ImportError:
        return
    if getattr(MMDistributedDataParallel._run_ddp_forward, "_compat_patched", False):
        return

    def _run_ddp_forward(self, *inputs, **kwargs):
        if self.device_ids:
            inputs, kwargs = self.to_kwargs(inputs, kwargs, self.device_ids[0])
            return self.module(*inputs[0], **kwargs[0])
        return self.module(*inputs, **kwargs)

    _run_ddp_forward._compat_patched = True
    MMDistributedDataParallel._run_ddp_forward = _run_ddp_forward


def main() -> None:
    dist.init = _init_from_torchrun_env
    context.init = _init_from_torchrun_env
    _patch_yapf()
    _patch_mmcv_get_stream()
    _patch_mmcv_ddp_forward()

    # BEVFUSION_ENTRY=test.py reuses this shim for tools/test.py (evaluation only).
    entry_name = os.environ.get("BEVFUSION_ENTRY", "train.py")
    entry = os.path.join(os.path.dirname(os.path.abspath(__file__)), entry_name)
    sys.argv = [entry] + sys.argv[1:]
    runpy.run_path(entry, run_name="__main__")


if __name__ == "__main__":
    main()
