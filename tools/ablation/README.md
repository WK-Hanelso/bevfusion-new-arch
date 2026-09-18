# Phase 2 ablation config family

이 계열은 학습 코드를 바꾸지 않고 config 이름만 선택해 LiDAR encoder,
camera view transform, fuser, object head를 비교한다. 모든 leaf는 nuScenes
6-camera, single sweep, GT-Aug 미사용, CBGS, 256×704 입력, 동일 augmentation,
AdamW/cyclic schedule, 20 epochs를 공유한다.

## 전체 16개 조합

파일명의 네 bit는 차례로 DSVT, WidthFormer, GFusion, DAL 사용 여부다.

| canonical config | LiDAR | camera vtransform | fuser | head |
|---|---|---|---|---|
| `dsvt0_wf0_gf0_dal0.yaml` | SparseEncoder | DepthLSS | ConvFuser | TransFusion |
| `dsvt0_wf0_gf0_dal1.yaml` | SparseEncoder | DepthLSS | ConvFuser | DAL |
| `dsvt0_wf0_gf1_dal0.yaml` | SparseEncoder | DepthLSS | DepthGFusion | TransFusion |
| `dsvt0_wf0_gf1_dal1.yaml` | SparseEncoder | DepthLSS | DepthGFusion | DAL |
| `dsvt0_wf1_gf0_dal0.yaml` | SparseEncoder | WidthFormer | ConvFuser | TransFusion |
| `dsvt0_wf1_gf0_dal1.yaml` | SparseEncoder | WidthFormer | ConvFuser | DAL |
| `dsvt0_wf1_gf1_dal0.yaml` | SparseEncoder | WidthFormer | DepthGFusion | TransFusion |
| `dsvt0_wf1_gf1_dal1.yaml` | SparseEncoder | WidthFormer | DepthGFusion | DAL |
| `dsvt1_wf0_gf0_dal0.yaml` | DSVT | DepthLSS | ConvFuser | TransFusion |
| `dsvt1_wf0_gf0_dal1.yaml` | DSVT | DepthLSS | ConvFuser | DAL |
| `dsvt1_wf0_gf1_dal0.yaml` | DSVT | DepthLSS | DepthGFusion | TransFusion |
| `dsvt1_wf0_gf1_dal1.yaml` | DSVT | DepthLSS | DepthGFusion | DAL |
| `dsvt1_wf1_gf0_dal0.yaml` | DSVT | WidthFormer | ConvFuser | TransFusion |
| `dsvt1_wf1_gf0_dal1.yaml` | DSVT | WidthFormer | ConvFuser | DAL |
| `dsvt1_wf1_gf1_dal0.yaml` | DSVT | WidthFormer | DepthGFusion | TransFusion |
| `dsvt1_wf1_gf1_dal1.yaml` | DSVT | WidthFormer | DepthGFusion | DAL |

## 기존 이름 별칭

별칭은 대응 canonical과 YAML 내용이 같으며, 생성된 헤더에 canonical 이름을
기록한다.

| alias | canonical |
|---|---|
| `b0_legacy.yaml` | `dsvt0_wf0_gf0_dal0.yaml` |
| `e1_dsvt.yaml` | `dsvt1_wf0_gf0_dal0.yaml` |
| `e2_widthformer.yaml` | `dsvt0_wf1_gf0_dal0.yaml` |
| `e3_gfusion.yaml` | `dsvt0_wf0_gf1_dal0.yaml` |
| `e4_dal.yaml` | `dsvt0_wf0_gf0_dal1.yaml` |
| `c2_dsvt_widthformer.yaml` | `dsvt1_wf1_gf0_dal0.yaml` |
| `c3_dsvt_widthformer_gfusion.yaml` | `dsvt1_wf1_gf1_dal0.yaml` |
| `c4_full.yaml` | `dsvt1_wf1_gf1_dal1.yaml` |

## 런처 실험 ID

`experiments.py`가 런처·생성기·집계기의 단일 실험 registry다. bit 순서는
DSVT, WidthFormer, GFusion, DAL이다. 대문자 실험 ID와 위의 소문자 legacy
config 별칭은 서로 다른 이름 체계이므로 혼용하지 않는다.

| ID | bits | canonical config |
|---|---:|---|
| B0 | 0000 | `dsvt0_wf0_gf0_dal0.yaml` |
| A1 | 1000 | `dsvt1_wf0_gf0_dal0.yaml` |
| A2 | 0100 | `dsvt0_wf1_gf0_dal0.yaml` |
| A3 | 0001 | `dsvt0_wf0_gf0_dal1.yaml` |
| A4 | 0010 | `dsvt0_wf0_gf1_dal0.yaml` |
| C1 | 1100 | `dsvt1_wf1_gf0_dal0.yaml` |
| C2 | 1001 | `dsvt1_wf0_gf0_dal1.yaml` |
| C3 | 1010 | `dsvt1_wf0_gf1_dal0.yaml` |
| C4 | 0101 | `dsvt0_wf1_gf0_dal1.yaml` |
| C5 | 0110 | `dsvt0_wf1_gf1_dal0.yaml` |
| C6 | 0011 | `dsvt0_wf0_gf1_dal1.yaml` |
| P1 | 1101 | `dsvt1_wf1_gf0_dal1.yaml` |
| P2 | 1110 | `dsvt1_wf1_gf1_dal0.yaml` |
| P3 | 1011 | `dsvt1_wf0_gf1_dal1.yaml` |
| P4 | 0111 | `dsvt0_wf1_gf1_dal1.yaml` |
| FINAL | 1111 | `dsvt1_wf1_gf1_dal1.yaml` |

공통 B0는 ResNet-34 + LSSFPN(`in_indices: [2,1]`, stride 16),
DepthLSSTransform, voxel 0.075의 SparseEncoder, ConvFuser,
TransFusionHead(`out_size_factor: 8`)다. 원 BEVFusion camera+LiDAR 기준은
ResNet-50 + GeneralizedLSSFPN(stride 8)이므로 B0는 그 config의 단순 별칭이
아니다. camera backbone과 실험 조건을 신규 edge target에 맞춰 고정한 비교
기준이다.

DSVT leaf는 voxel 0.3, `out_size_factor: 2`, grid 360×360×1을 함께 선택한다.
모든 encoder/fuser 출력은 180×180이고 decoder 입력은 256 channels다.

## 학습

일반 형식은 다음과 같다.

```bash
BEVFUSION_CONFIG=configs/nuscenes/det/ablation/dsvt0_wf0_gf0_dal0.yaml
torchpack dist-run -np 1 python tools/train.py "$BEVFUSION_CONFIG" \
  --run-dir runs/ablation/dsvt0_wf0_gf0_dal0
```

`dsvt0_*` 조합은 config와 run directory를 바꿔 같은 명령을 사용한다.
`dsvt1_*` 조합은 변환된 공식 checkpoint를 명시적으로 로드한다. leaf에는
`load_from` 기본값이 없다.

```bash
BEVFUSION_CONFIG=configs/nuscenes/det/ablation/dsvt1_wf0_gf0_dal0.yaml
torchpack dist-run -np 1 python tools/train.py \
  "$BEVFUSION_CONFIG" \
  --run-dir runs/ablation/dsvt1_wf0_gf0_dal0 \
  --load_from pretrained/dsvt_nuscenes_official_lidar.pth
```

공식 checkpoint 생성과 적용 범위는
[`tools/dsvt_pretrained/README.md`](../dsvt_pretrained/README.md)를 따른다.

## 2단계 wave 실행

런처는 `bevfusion-b200` conda 환경에서 `torchrun`을 실행한다. screening의
기본값은 16개 전체, 6 epoch, 작업당 2 GPU, 4개 병렬이다. 먼저 실제 파일을
만들지 않는 dry-run으로 배정, config, override와 DSVT checkpoint 적용 범위를
확인한다.

```bash
python tools/ablation/launch_waves.py \
  --phase screening --dry-run \
  --dataroot /data/nuscenes \
  --load-from-dsvt /weights/dsvt_nuscenes_official_lidar.pth

python tools/ablation/launch_waves.py \
  --phase screening \
  --dataroot /data/nuscenes \
  --load-from-dsvt /weights/dsvt_nuscenes_official_lidar.pth \
  --seed 0
```

`--load-from-dsvt`는 첫 bit가 1인 8개 실험에만 `--load_from`으로 전달된다.
각 slot은 고정된 `CUDA_VISIBLE_DEVICES`와 별도 master port를 사용한다. 한
wave의 작업이 모두 끝나면 다음 wave가 시작된다. 비정상 exit 또는 로그의
NaN/Inf, CUDA error, OOM을 실패로 기록하며, 실행 중 발견하면 해당 slot의
프로세스만 종료한다. 실패 작업을 자동 재시작하지 않고 다른 slot과 다음
wave는 계속 실행한다.

`--epochs`는 Torchpack의 `--max_epochs` override로 전달된다. recursive config의
`${max_epochs}` 때문에 `runner.max_epochs`와 GridMask가 함께 바뀌며, MMCV
cyclic LR/momentum hook은 runner의 변경된 `max_iters`를 전체 cycle 길이로
사용한다. 원본 YAML은 수정하지 않는다.

중단 뒤 orchestration을 다시 시작할 때 `--resume`을 추가하면
`metrics.json`의 status가 `completed`인 ID만 건너뛴다. 이는 checkpoint
`--resume_from`이 아니며 실패 run을 자동 재개하지 않는다.

screening 집계와 Top-4 출력은 다음과 같다. CSV는 epoch별 행을 담고 Markdown은
선택 지표 기준 각 ID의 best epoch 및 B0 대비 delta를 담는다.

```bash
python tools/ablation/aggregate.py \
  --phase screening --top-k 4 --by NDS \
  --force-include-b0-final
# 마지막 줄 예: TOP_K_IDS=B0 FINAL A1 C3
```

Top-4 full 학습은 ID를 명시한다. final 기본값은 20 epoch, 작업당 4 GPU,
2개 병렬이다.

```bash
python tools/ablation/launch_waves.py \
  --phase final --ids B0 FINAL A1 C3 --dry-run \
  --dataroot /data/nuscenes \
  --load-from-dsvt /weights/dsvt_nuscenes_official_lidar.pth

python tools/ablation/launch_waves.py \
  --phase final --ids B0 FINAL A1 C3 \
  --dataroot /data/nuscenes \
  --load-from-dsvt /weights/dsvt_nuscenes_official_lidar.pth

python tools/ablation/aggregate.py --phase final --top-k 4 --by NDS
```

### 산출물

각 실행은 `experiments/runs/<phase>/<ID>/`에 다음을 만든다.

| 경로 | 내용 |
|---|---|
| `config.yaml` | Torchpack override까지 반영되어 trainer가 dump한 resolved config |
| `command.txt` | GPU 배정을 포함한 재현 명령 |
| `env.txt` | commit, seed, phase, bit, host와 GPU 배정 |
| `train.log` | stdout/stderr 통합 학습·평가 로그 |
| `metrics.json` | running/completed/failed 상태, exit code, wall time, 실패 근거 |
| `checkpoints/` | epoch/latest/best checkpoint 출력 위치 |

phase 디렉터리에는 시작 시 캡처한 `env_nvidia_smi.txt`, `env_nvcc.txt`,
`env_pip_freeze.txt`가 생긴다. 세 명령 모두 `conda run -n bevfusion-b200`에서
캡처한다. 집계 결과는 `results/screening_summary.{csv,md}` 또는
`results/final_ablation.{csv,md}`다. CSV에는 mAP/NDS/mATE/mASE/mAOE/mAVE/mAAE,
s/iter, run peak VRAM, wall time, NaN, checkpoint, commit과 seed가 포함된다.

## 생성·검사와 server smoke

leaf는 직접 편집하지 않고 생성기로 갱신한다.

```bash
python tools/ablation/gen_configs.py
python tools/ablation/gen_configs.py --check
pytest -q tools/ablation/tests
bash -n tools/ablation/smoke_all.sh
python tools/ablation/launch_waves.py --phase screening --dry-run
```

실제 1-step forward/backward는 CUDA, mmcv, torchpack, nuScenes data가 있는
서버에서 실행한다. canonical config 16개를 순차 실행하고 결과를
`tools/ablation/smoke_results.md`에 append한다.

```bash
DATAROOT=/data/nuscenes DEVICE=0 bash tools/ablation/smoke_all.sh
```

## 단일 센서 dispatch 계약

`ModularBEVFusion`은 feature를 `camera`, `lidar` key로 보관하며 두 센서가
있을 때만 선택된 fuser를 호출한다. 단일 센서는 fuser를 우회한다. DAL은
LiDAR가 있으면 encoder의 LiDAR BEV를 회귀 입력으로 사용하고, camera-only면
decoder 출력을 회귀 입력 대체값으로 사용한다. 표준 TransFusion head는 항상
decoder 출력과 metadata만 받는다.
