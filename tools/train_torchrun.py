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


def main() -> None:
    dist.init = _init_from_torchrun_env
    context.init = _init_from_torchrun_env
    _patch_yapf()

    entry = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")
    sys.argv = [entry] + sys.argv[1:]
    runpy.run_path(entry, run_name="__main__")


if __name__ == "__main__":
    main()
