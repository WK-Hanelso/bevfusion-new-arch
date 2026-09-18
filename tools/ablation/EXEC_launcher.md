# EXEC — ablation wave launcher and aggregation

- Date: 2026-09-19 (Asia/Seoul)
- Branch: `b200-env`
- Scope: `SPEC_launcher.md` requirements 1–6
- Constraint: CPU host, torch 1.12, no mmcv/torchpack; no packages installed and no
  GPU training performed

## Implemented

1. Added `experiments.py` as the single 16-entry ID/bit/canonical-config registry.
   `gen_configs.py`, the launcher, and the aggregator import it. The eight legacy
   generated config filenames remain byte-stable.
2. Added `launch_waves.py` with phase-dependent defaults, ID selection, contiguous
   GPU groups, per-slot torchrun ports, wave barriers, DSVT-only checkpoint loading,
   dry-run, and completed-run skipping. Every real run records command/environment/
   status/log/checkpoints, and copies the trainer's resolved `configs.yaml` to
   `config.yaml`.
3. `--epochs` is passed as Torchpack `--max_epochs`. In this config hierarchy,
   `runner.max_epochs` and GridMask reference `${max_epochs}`; MMCV cyclic LR and
   momentum hooks derive their duration from the runner's resulting `max_iters`.
   No generated YAML is edited to shorten screening.
4. Added `aggregate.py`. It parses per-epoch nuScenes mAP, NDS, five TP errors,
   mean s/iter, run peak logged memory, wall time, NaN, checkpoint, commit, and seed.
   It writes the required phase CSV/Markdown files and emits a shell-readable
   `TOP_K_IDS=...` line. Optional forced B0/FINAL inclusion still keeps the requested
   Top-K cardinality.
5. Real launcher starts capture `nvidia-smi`, `nvcc --version`, and `pip freeze`
   through `conda run -n bevfusion-b200` in the phase directory.
6. Added CPU-only launcher tests and a representative MMCV/mmdet3d log fixture with
   two epochs, all requested detection metrics, iteration timing, memory, and NaN.
   Updated `README.md` with screening/final commands and artifact semantics.

Failure policy follows the spec: nonzero exit or fatal log patterns (NaN, Inf, CUDA
error/status, OOM) produce `status: failed`; fatal patterns stop only that slot. There
is no automatic experiment restart. Other slots complete, then the next wave starts.

## Verification

Commands run from repository root:

```text
python -m py_compile tools/ablation/experiments.py tools/ablation/gen_configs.py tools/ablation/launch_waves.py tools/ablation/aggregate.py
python tools/ablation/gen_configs.py --check
pytest -q tools/ablation/tests
bash -n tools/ablation/smoke_all.sh
python tools/ablation/launch_waves.py --phase screening --dry-run --dataroot /fixture/nuscenes --load-from-dsvt /fixture/dsvt.pth
python tools/ablation/launch_waves.py --phase final --ids B0 FINAL A1 C3 --dry-run
git diff --check
```

Observed results:

```text
checked 16 canonical and 8 alias ablation leaf configs
80 passed in 7.92s
screening plans: 16
screening DSVT load overrides: 8
final plans: 4
final defaults: epochs=20, gpus_per_job=4, parallel=2
git diff --check: clean
```

The screening dry-run created no run artifacts. CUDA execution, conda environment
capture, resolved Torchpack config generation, and real mmdet3d evaluation remain
server-side checks because the local environment intentionally lacks those runtime
dependencies.
