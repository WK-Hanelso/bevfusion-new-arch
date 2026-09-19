# 검증 매트릭스 M1~M7 설계와 실행 기록

## 범위

`verify_matrix.sh`는 긴 screening/final 실행 전에 M1~M7을 한 번에 수행한다.
M8(spconv2)은 요청 범위에서 제외했으며 `--with-spconv2`를 넘기면 오류로
종료한다. 소스 변경은 `tools/ablation/` 아래의 이 문서와 스크립트뿐이다.

실제 GPU 학습은 이 작업 환경에서 실행하지 않았다. 아래의 로컬 검증은 Bash
문법, 기존 ablation unit test, dry-run 명령 계획만 대상으로 한다. 실제 PASS
마커는 서버에서 M1~M7이 모두 통과한 경우에만 생성된다.

## 실행 설계

한 실행의 모든 로그는 기존 결과와 섞이지 않도록 다음의 새 디렉터리에 둔다.

```text
experiments/verify/run_<rev8>_<UTC>_<pid>/
├── mini_nuscenes/        # train 512개, val info는 원본 전체를 symlink
├── m1_runs/screening/    # aggregate.py의 현재 screening fallback 구조
├── m2/ ... m7/
└── results/              # M7 CSV/Markdown
```

성공 마커는 현재 full revision을 사용한
`experiments/verify/PASS_<rev>.txt`다. 새 실행 디렉터리를 사용하므로 이전 실행의
checkpoint, 로그 또는 RESULT가 새로운 PASS 판정에 포함되지 않는다.
같은 revision의 기존 PASS 마커가 있으면 새 실제 실행 시작 시 실행 디렉터리의
`PREVIOUS_PASS_<rev>.txt`로 보존 이동한다. 따라서 새 실행이 실패하는 동안 stale
마커가 screening을 통과시키지 않는다.

| 항목 | 배치와 명령 구성 | PASS 판정 |
|---|---|---|
| M1 | registry의 16개 ID를 8개씩 2라운드, GPU당 1개 config. `--max_epochs 2`, `samples_per_gpu=16`, `GradientCumulativeOptimizerHook`, `cumulative_iters=2` | 각 ID exit 0, `epoch_2.pth`, `mAP:` 2회 이상, loss 필드 존재, loss/grad norm NaN·Inf 없음 |
| M2 | `B0 A1 A2 A3 A4 C5 P2 FINAL` 8개를 GPU당 1개씩 M1 `epoch_1.pth`에서 resume. `--max_epochs 2` | exit 0, 로그에 epoch 1 resume 근거, `epoch_2.pth`, loss NaN·Inf 없음 |
| M3 | B0는 GPU 0~3, FINAL은 GPU 4~7에서 동시에 실행. 4 process, `samples_per_gpu=8`, 1 epoch | 두 ID 모두 exit 0, `epoch_1.pth`, 평가 mAP, finite loss |
| M4 | B0와 FINAL을 각각 2 process, `samples_per_gpu=16`, 1 epoch | 두 ID 모두 exit 0, `epoch_1.pth`, 평가 mAP, finite loss |
| M5 | `BEVFUSION_ENTRY=test.py`와 `tools/train_torchrun.py`로 B0 full val 평가 | exit 0 및 `mAP:` 출력 |
| M6 | `PYTHONPATH=. python -m pytest -q tools/ablation/tests` | exit 0 |
| M7 | aggregate가 `verify` phase를 지원하면 이를 사용하고, 현재처럼 없으면 M1 디렉터리를 `--runs-root`로 주어 `screening` phase 실행 | exit 0, CSV/MD 생성, 비어 있지 않은 `TOP_K_IDS=` 출력 |

M5의 `tools/test.py`는 checkpoint positional 인자를 필수로 받고 무조건
`load_checkpoint()`를 호출한다. `mmdet3d/` 및 `tools/test.py`는 수정 금지
범위이므로 checkpoint 없는 랜덤 초기화 평가를 만들 수 없다. SPEC이 허용한
fallback대로 M1 B0의 `epoch_1.pth`를 사용한다. 평가 데이터 경로는 mini root의
full-val symlink와 `data.test.dataset_root`, `data.test.ann_file` override로
고정한다.

DSVT bit가 1인 M1/M3/M4 작업에는
`pretrained/dsvt_nuscenes_official_lidar.pth`를 `--load_from`으로 전달한다.
M2는 optimizer/epoch/model 상태를 담은 `--resume_from`만 사용한다. 모든 학습
명령은 다음 형태다.

```text
CUDA_VISIBLE_DEVICES=... conda run -n bevfusion-b200 --no-capture-output \
  torchrun --master_port=P --nproc_per_node=N tools/train_torchrun.py <cfg> ...
```

GPU 작업은 별도 process group으로 실행한다. Ctrl-C/TERM이면 활성 wrapper가
각 process group을 종료한다. 실행 중인 배치는 기본 60초마다 ID별 RUNNING/DONE,
최근 epoch·평가·오류 줄과 `nvidia-smi` 요약을 출력한다. 개별 작업 실패는 같은
배치의 다른 작업을 중단시키지 않으며 마지막 표에서 함께 보고한다.

## 서버 실행 방법

먼저 명령과 GPU 배정을 확인한다. dry-run은 nuScenes 경로와 checkpoint의
존재를 검사하지 않고 파일도 만들지 않는다.

```bash
bash tools/ablation/verify_matrix.sh /data/nuscenes --dry-run
```

실제 실행:

```bash
LOAD_FROM_DSVT=/absolute/path/dsvt_nuscenes_official_lidar.pth \
  bash tools/ablation/verify_matrix.sh /data/nuscenes
```

기본 GPU는 `0,1,2,3,4,5,6,7`이다. 다른 물리 GPU 번호를 쓸 때도 정확히 8개를
중복 없이 지정한다.

```bash
VERIFY_GPUS=2,3,4,5,6,7,8,9 \
LOAD_FROM_DSVT=/absolute/path/dsvt_nuscenes_official_lidar.pth \
  bash tools/ablation/verify_matrix.sh /data/nuscenes
```

지원 환경 변수는 다음과 같다.

| 변수 | 기본값 | 의미 |
|---|---|---|
| `VERIFY_GPUS` | `0,1,2,3,4,5,6,7` | 사용할 8개 GPU |
| `CONDA_ENV` | `bevfusion-b200` | conda 환경 |
| `LOAD_FROM_DSVT` | `pretrained/dsvt_nuscenes_official_lidar.pth` | DSVT 초기화 checkpoint |
| `VERIFY_TRAIN_SAMPLES` | `512` | mini train info 개수 |
| `VERIFY_STATUS_SECS` | `60` | 상태 출력 간격 |
| `VERIFY_MASTER_PORT_BASE` | `31600` | torchrun port 기준값 |

마지막 Markdown 표의 M1~M7이 전부 PASS일 때만 마커가 기록된다. 실패 시 각
`RESULT`, `train.log`, M6 `pytest.log`, M7 `aggregate.log`를 확인한다.

## 이 환경에서 수행한 검증 (2026-09-19)

### Bash 문법

```bash
bash -n tools/ablation/verify_matrix.sh
```

결과: PASS.

### dry-run

존재하지 않는 데이터 경로를 전달해 실제 데이터 접근이나 파일 생성 없이 명령
계획만 검사했다.

```bash
bash tools/ablation/verify_matrix.sh /path/not/required/in/dry-run --dry-run
```

출력에서 확인한 작업 수:

```text
M1=16 M2=8 M3=2 M4=2 M5=1 M6=1 M7=1
```

M1 명령은 모두 `--max_epochs 2`, `samples_per_gpu 16`, cumulative hook 2를,
M2는 `--resume_from`, M3은 4 process/batch 8, M4는 2 process/batch 16을
포함했다. M5는 `BEVFUSION_ENTRY=test.py`, M6은 지정 pytest 명령, M7은 현재
aggregate CLI에 맞는 screening fallback을 출력했다. dry-run 전후 대상 임시
디렉터리의 추가 파일 수는 0이었다.

### ablation unit test

```bash
PYTHONPATH=. python -m pytest -q tools/ablation/tests
```

결과: `88 passed in 7.59s`.

### shellcheck와 GPU 실행

현재 셸에는 `shellcheck` 실행 파일이 없어 실행하지 못했고 패키지도 설치하지
않았다. 실제 CUDA/nuScenes M1~M7 역시 요청대로 서버 실행 대상으로 남겼다.
따라서 이 문서에는 GPU PASS나 성능 수치를 기록하지 않는다.
