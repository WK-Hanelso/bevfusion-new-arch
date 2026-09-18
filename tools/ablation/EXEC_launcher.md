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

## §7 풀 스케줄러 (2026-09-19)

### 구현

1. Wave 배리어를 GPU 그룹 풀로 교체했다. 선택된 ID 순서의 단일 대기열을
   유지하고, 완료·실패·fatal 종료로 slot이 비면 다음 ID를 즉시 같은 slot에
   투입한다. `[PLAN]`은 전체 대기열에 대해 한 번씩 출력하고 실제 투입과
   종료는 `[START]`, `[COMPLETED]`/`[FAILED]`로 즉시 출력한다.
2. `--gpus`(기본 `0,1,2,3,4,5,6,7`)를 추가했다. GPU 목록은
   `--gpus-per-job` 크기의 그룹으로 분할하고 `--parallel`은 그룹 수 상한으로만
   적용한다. 생략한 `--parallel`은 실제 그룹 수를 사용한다. slot별 master
   port는 `29500 + slot`이다.
3. 이전 slot 프로세스가 종료되고 `wait()`로 회수된 뒤에만 같은 port로 다음
   작업을 시작한다. Fatal 패턴으로 종료 요청한 프로세스는 다른 slot을 막지
   않고 2초 간격, 최대 5회 확인·종료 요청 후 필요하면 kill한다.
4. `metrics.json`의 기존 필드와 상태 값은 유지했다. `wave`는 실제 투입 순번을
   기록한다. 실행 중 child PID는 `launcher.pid`에 기록하고 정상/실패 종료 시
   삭제한다. `--resume`은 completed와 살아 있는 running PID를 건너뛰며,
   PID가 없거나 죽은 stale running 작업은 다시 대기열에 넣는다.
5. 기존 80개 테스트에 풀 refill, GPU 그룹 분할, dry-run 형식, running PID
   resume 검증 4개를 추가했다. 가짜 subprocess 테스트에서 B0가 즉시 실패한
   뒤 네 번째 작업 A3가 다른 두 장기 작업을 기다리지 않고 slot 0에 투입됨을
   확인했다.

### 검증

저장소 루트에서 CPU-only 환경(torch/mmcv 및 패키지 설치 없음)으로 실행했다.

```text
PYTHONPATH=. python -m pytest -q tools/ablation/tests
python tools/ablation/launch_waves.py --phase screening --dry-run --gpus 0,1,2,3,6,7
python tools/ablation/launch_waves.py --phase screening --dry-run
git diff --check
```

관측 결과:

```text
84 passed in 7.52s
custom GPU dry-run: exit 0, 16 [PLAN], parallel=3, groups=0,1 / 2,3 / 6,7
default GPU dry-run: exit 0, 16 [PLAN], parallel=4, groups=0,1 / 2,3 / 4,5 / 6,7
git diff --check: clean
```

두 dry-run 모두 파일을 생성하지 않았고, screening 기본 6 epochs와 작업당
2 GPU, slot별 `--master_port`, 16개 기존 실험 순서를 유지했다. 실제 CUDA
학습 및 conda 환경 캡처는 이 호스트의 의존성 제약 때문에 수행하지 않았다.
