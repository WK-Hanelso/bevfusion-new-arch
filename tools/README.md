# 학습·재개·평가 가이드

[저장소](../README.md) · [모델 아키텍처](../new-arch.md) · [Docker 환경](../docker/README.md) · [PTH → ONNX](../deployment/README.md)

기존 BEVFusion 3/6-camera와 신규 DynamicBEVFusion의 학습 절차를 이 문서에서 관리한다. 별도 training 프로젝트는 만들지 않는다. 아래 명령은 기존 `configs/`, `mmdet3d/`, `tools/`를 사용하며 모든 상대 경로는 repository root 기준이다.

## 1. 환경과 실행 파일

Docker 설치·이미지 빌드·GPU/데이터/cache mount는 [docker/README.md](../docker/README.md)를 먼저 따른다. 컨테이너 안에서는 `/workspace/bevfusion`에서 실행하며 별도 venv activate가 필요 없다. 준비된 host venv를 쓰는 경우에만 다음을 실행한다.

```bash
cd /home/culee/workspace/bevfusion
source /home/culee/2608_bevfusion/bin/activate
```

| 파일 | 역할 |
|---|---|
| `tools/data_converter/nuscenes_converter.py` | raw nuScenes → train/val info PKL |
| `tools/data_converter/create_gt_database.py` | 기존 ObjectPaste용 GT database 생성 |
| `tools/create_data.py` | 기존 데이터 준비 CLI; 아래 full split 주의사항 참조 |
| `tools/smoke_train.py` | 실제 1 batch forward/backward/optimizer 검사, checkpoint 저장 없음 |
| `tools/train.py` | 기존 Torchpack 학습 진입점 |
| `tools/train_torchrun.py` | 같은 train.py를 torchrun 환경으로 호출하는 wrapper |
| `mmdet3d/apis/train.py` | optimizer/runner, validation hook, resume/load 연결 |
| `tools/test.py` | checkpoint 평가·결과 저장; Torchpack/MPI 환경 필요 |

## 2. nuScenes-mini / full 데이터 준비

학습 기본 경로는 `data/nuscenes/`다. 현재 host의 mini 원본은 `/home/culee/nuscenes_mini`이며 Docker 기본 mount가 이를 위 경로에 제공한다. Full은 별도 데이터 root를 준비해 mount source를 변경한다. Mini/full의 info PKL 파일명이 같으므로 서로 다른 root로 관리하고 기존 PKL/GT database를 덮어쓰지 않는다.

```text
data/nuscenes/
  samples/                       LiDAR + 선택할 camera 원본
  sweeps/                        이전 LiDAR sweeps
  v1.0-mini/ 또는 v1.0-trainval/   nuScenes metadata JSON
  maps/                          devkit 및 기존 map-label pipeline에 필요한 자료
  nuscenes_infos_train.pkl
  nuscenes_infos_val.pkl
  nuscenes_dbinfos_train.pkl      ObjectPaste 사용 시
  nuscenes_gt_database/           ObjectPaste 사용 시
```

기존 3/6-camera config는 multisweep·ObjectPaste·map label loader를 사용한다. 신규 config는 single-sweep이며 ObjectPaste와 map label loader를 사용하지 않는다. 신규 모델의 다섯 번째 point feature는 ring index이고 기존 multisweep의 time lag와 동일하지 않으므로 입력 column을 임의로 혼용하지 않는다.

이미 준비된 PKL이 있으면 아래 생성을 생략한다. PKL은 신뢰할 수 있는 출처의 파일만 로드한다. 학습·평가에는 PKL의 `metadata.version`, 실제 metadata 디렉터리, sample 경로가 일치해야 한다. Host 절대 경로로 생성한 PKL은 Docker에 자동 맞춤되지 않는다. 가급적 실행 환경의 root에서 상대 경로 `data/nuscenes`로 생성하고, 학습 때도 같은 상대 경로를 제공한다.

### Info PKL 생성

다음 명령은 **데이터 root에 쓰기 권한이 있는 준비 환경**에서만 실행한다. Docker 학습 기본 mount는 read-only이므로 준비용 컨테이너에서만 해당 데이터 mount의 `:ro`를 제거하고, 준비 후 학습용 read-only mount로 돌아간다. 이 명령은 데이터를 다운로드하지 않는다.

```bash
# mini: raw samples/sweeps/metadata가 준비되어 있고 기존 PKL이 없을 때
python -c "from tools.data_converter.nuscenes_converter import create_nuscenes_infos; create_nuscenes_infos('data/nuscenes', 'nuscenes', version='v1.0-mini', max_sweeps=10)"
```

Full train/val은 별도 root의 `v1.0-trainval` metadata와 해당 samples/sweeps를 준비하고 위 명령의 `version`만 `v1.0-trainval`로 바꾼다. 실제 읽는 sweeps 수는 모델 config가 결정한다. Converter는 6-camera 정보를 만들고 dataset이 config 순서로 필요한 camera를 선택한다.

기존 함수의 단계별 호출을 사용하는 이유: `tools/create_data.py nuscenes --version v1.0`은 trainval 다음에 test split도 준비하며 GT database 생성까지 호출한다. Trainval만 가진 환경에서는 test metadata에서 실패할 수 있고, test 단계에서도 train PKL을 참조하므로 trainval 준비용 단일 명령으로 권장하지 않는다. 이 문서에서는 해당 CLI를 변경하거나 임의로 오류를 무시하지 않는다.

### GT database 생성 — 기존 ObjectPaste 사용 시

```bash
python -c "from tools.data_converter.create_gt_database import create_groundtruth_database; create_groundtruth_database('NuScenesDataset', 'data/nuscenes', 'nuscenes', 'data/nuscenes/nuscenes_infos_train.pkl')"
```

위 함수는 `nuscenes_gt_database/`와 `nuscenes_dbinfos_train.pkl`을 생성하며 기본적으로 multisweep 입력을 읽는다. 신규 config에는 필요하지 않다. 기존 config의 map-expansion 자료 누락은 GT database 생성과 별개 문제다.

## 3. Config 선택과 smoke

아키텍처와 camera 순서는 [new-arch.md](../new-arch.md)를 기준으로 한다. 다음 중 사용할 YAML 하나를 선택한다.

| 선택 | config |
|---|---|
| 기존 전방 3-camera | `configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50/convfuser.yaml` |
| 기존 6-camera | `configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50/convfuser_6cam.yaml` |
| 신규 6-camera | `configs/nuscenes/det/transfusion/secfpn/lidar/dsvt_dgf_dal_widthformer_0p3.yaml` |

```bash
# 이후 명령에서 같은 shell 변수 사용; 다른 모델은 이 값만 선택한 YAML로 변경
BEVFUSION_CONFIG=configs/nuscenes/det/transfusion/secfpn/lidar/dsvt_dgf_dal_widthformer_0p3.yaml
python tools/smoke_train.py "$BEVFUSION_CONFIG" \
  --dataroot data/nuscenes --device 0
```

Host의 mini 경로를 직접 쓰려면 smoke의 `--dataroot /home/culee/nuscenes_mini`를 사용한다. 이 옵션은 smoke에만 적용되며 전체 학습의 dataset root를 바꾸지 않는다.

Smoke는 batch=1/worker=0으로 실제 한 batch를 읽고 loss 유한성·branch gradient·optimizer step을 확인한다. 기본은 FP32이며 전체 trainer의 `model.init_weights()`나 checkpoint 저장·validation hook을 실행하지 않는다. 따라서 smoke PASS가 pretrained 다운로드, 전체 학습, 재개 또는 평가 성공을 보장하지 않는다.

| smoke 옵션 | 의미 |
|---|---|
| `--forward-only` | loss forward만 실행, backward/optimizer 생략 |
| `--fp16` | MMCV mixed-precision wrapping 사용; 전체 trainer hook과 동일 검증은 아님 |
| `--object-only` | map label loader만 제외; GT database 요구는 그대로 |

## 4. 전체 학습

### 단일 GPU

```bash
python -m torch.distributed.run --nproc_per_node=1 tools/train_torchrun.py \
  "$BEVFUSION_CONFIG" --run-dir runs/new-arch \
  --data.samples_per_gpu 1 --data.workers_per_gpu 0
```

위 batch/worker 값은 시작용 예시이며 config 기본값(각각 4/4)을 명시적으로 낮춘다. Epoch 기본값은 기존 3/6-camera가 6, 신규가 20이다. 세 config 모두 기본 FP16 optimizer hook을 상속한다. FP32 진단은 `--fp16 None`을 추가하며 smoke 기본 precision과 혼동하지 않는다.

`train.py`는 epoch별 validation도 구성하므로 train PKL뿐 아니라 val PKL과 평가용 metadata가 필요하다. 서로 다른 실험·모델은 다른 `--run-dir`를 사용한다.

### 단일 노드 멀티 GPU

```bash
# 이 예시는 컨테이너/host에 GPU 두 개가 노출된 경우
python -m torch.distributed.run --nproc_per_node=2 tools/train_torchrun.py \
  "$BEVFUSION_CONFIG" --run-dir runs/new-arch-2gpu \
  --data.samples_per_gpu 1 --data.workers_per_gpu 2
```

`--nproc_per_node`는 노출한 GPU 수와 맞춘다. Gradient accumulation이 없으면 global batch는 GPU 수 × `samples_per_gpu`다. GPU 수 변경에 따른 learning rate 자동 보정을 가정하지 않는다. 멀티 노드 명령과 해당 구성의 성능 검증은 이 문서 범위 밖이다.

OpenMPI/mpi4py가 준비된 환경에서는 기존 `torchpack dist-run -np 1 python tools/train.py "$BEVFUSION_CONFIG" --run-dir runs/legacy-mpi`도 사용할 수 있다. 현재 host venv에는 mpi4py가 없으므로 학습은 위 torchrun wrapper를 사용한다.

### Config override

`train.py`는 YAML load → Torchpack override → recursive resolve 순서다. 추가 옵션은 `--key value` 형태이며 CLI에서 Python null 값은 `None`을 사용한다. 평가의 `--cfg-options key=value`와 문법이 다르다.

| 추가 옵션 예 | 의미 |
|---|---|
| `--dataset_root /absolute/path/nuscenes/` | 전체 학습 데이터 root; PKL 문자열 조합 때문에 마지막 `/` 유지 |
| `--max_epochs 30` | 총 목표 epoch, 추가 epoch 수가 아님 |
| `--optimizer.lr 0.0001` | learning rate 변경 |
| `--fp16 None` | FP16 hook 비활성화 |
| `--checkpoint_config.max_keep_ckpts 3` | 일반 epoch checkpoint 보존 개수 변경 |

## 5. 초기화와 학습 재개

| 설정 | 실제 동작 |
|---|---|
| `model.encoders.camera.backbone.init_cfg` 등 | `model.init_weights()`에서 backbone 초기화; Pretrained면 네트워크/cache 필요 가능 |
| `--load_from PATH` | 전체 모델 weight를 읽어 새 학습 시작; optimizer/epoch 재개 아님 |
| `--resume_from PATH` | checkpoint의 model/optimizer 및 epoch/iteration 복원 |

`resume_from`이 있으면 `load_from`보다 우선한다. `model.init_weights()`는 두 로딩보다 먼저 호출되므로 **학습 resume/load도 항상 오프라인이라고 가정하면 안 된다**. Exporter의 외부 pretrained 초기화 비활성화는 학습 trainer와 별도 기능이다. 로딩 경고와 model/config 일치를 확인한다. Trainer의 checkpoint 로딩은 exporter의 strict contract 검사와 동일하지 않다.

```bash
# 새 데이터/학습 실험을 기존 weight에서 시작
python -m torch.distributed.run --nproc_per_node=1 tools/train_torchrun.py \
  "$BEVFUSION_CONFIG" --run-dir runs/new-arch-finetune \
  --load_from /absolute/path/model.pth --resume_from None \
  --data.samples_per_gpu 1 --data.workers_per_gpu 0

# 기존 학습 이어서 수행: 원래 실험과 같은 config/precision/batch 설정 유지
python -m torch.distributed.run --nproc_per_node=1 tools/train_torchrun.py \
  "$BEVFUSION_CONFIG" --run-dir runs/new-arch \
  --resume_from runs/new-arch/latest.pth --load_from None \
  --data.samples_per_gpu 1 --data.workers_per_gpu 0
```

`latest.pth`는 실제 학습이 checkpoint를 저장한 뒤에만 생긴다. 이미 목표 epoch까지 완료한 checkpoint를 재개하려면 총 목표 `--max_epochs`를 명시적으로 늘려야 하며 스케줄 변경을 별도로 검토한다. Resume는 중단 지점의 RNG·dataloader까지 bitwise 동일하게 재현한다는 뜻이 아니다.

## 6. 로그·checkpoint와 평가

| `--run-dir` 아래 파일 | 용도 |
|---|---|
| `configs.yaml` | 실행 당시 override까지 반영한 resolved config |
| `<timestamp>.log`, `<timestamp>.log.json` | 학습 loss/validation 로그 |
| `tf_logs/` | TensorBoard event |
| `epoch_<N>.pth`, `latest.pth` | epoch checkpoint와 최신 checkpoint 링크 |
| `best_*.pth` | validation의 `object/nds` 개선 시 저장되는 best checkpoint |

기본 checkpoint hook은 매 epoch 저장하고 일반 epoch 파일은 최근 1개만 보존한다. 오래된 checkpoint가 필요하면 보존 개수를 미리 변경하거나 별도 보관한다. `latest.pth`가 정확도 기준 best인 것은 아니다.

### Checkpoint 평가

**OpenMPI/mpi4py가 준비된 환경에서 실행한다.** 학습 Dockerfile에는 이 의존성이 포함되어 있다. `tools/test.py`는 현재 `--launcher none`이어도 `dist.init()`를 호출하고 distributed 경로를 사용하므로 `python tools/test.py ...` 또는 plain torchrun만으로 되는 평가라고 안내하지 않는다. `tools/train_torchrun.py`는 평가 wrapper가 아니다. Docker 기본 예시처럼 non-root UID에서 실행하고, 평가용 torchrun wrapper는 현재 없다.

```bash
mkdir -p runs/new-arch/eval
torchpack dist-run -np 1 python tools/test.py \
  "$BEVFUSION_CONFIG" runs/new-arch/latest.pth \
  --eval bbox --out runs/new-arch/eval/predictions.pkl \
  --eval-options jsonfile_prefix=runs/new-arch/eval
```

사용할 checkpoint에 대응하는 config/학습 override를 평가에도 반영한다. `tools/test.py`는 resolve 후 `--cfg-options`를 적용하므로 `dataset_root` 하나만 변경해 이미 resolve된 nested 경로가 다시 계산된다고 가정하면 안 된다. 위 예시는 기본 `data/nuscenes/` mount 기준이다.

FP32 평가는 사용할 YAML에서 최상위 `fp16: null`을 명시한다. 원본을 보존하려면 같은 config 디렉터리에 평가용 YAML 사본을 두고 해당 항목만 변경한다. **MMCV 1.4.0의 `--cfg-options fp16=None`은 실제 None이 아닌 문자열로 파싱되므로 FP16 wrapping을 비활성화하지 못한다.** 학습의 `--fp16 None`과 혼용하지 않는다. Precision 변경은 평가 시 별도로 선택·기록한다.

평가는 PKL의 metadata version에 따라 `v1.0-mini → mini_val`, `v1.0-trainval → val`을 선택한다. `data.test.ann_file` 기본은 val PKL이다. `--out`만 지정하면 결과 저장이며 정확도 평가가 아니다. 위 명령은 predictions PKL과 `results_nusc.json`, `metrics_summary.json` 등 devkit 평가 결과를 남긴다. Mini 점수와 full val 점수는 동일 기준의 성능 수치로 비교하지 않는다.

## 7. Checkpoint 전달

학습 결과 공유 시 선택한 실제 PTH, `configs.yaml`, 원본 YAML과 부모 `default.yaml`들을 포함한 코드 revision, 학습 override, dataset version, 평가 결과·로그를 함께 전달한다. `latest.pth` 링크만 복사하면 대상 epoch 파일 없이 깨질 수 있다.

ONNX exporter에는 같은 모델 구조를 복원할 수 있는 config를 지정한다. `runs/.../configs.yaml`은 실행 기록이며 export용 YAML로 자동 호환된다고 가정하지 않는다. ONNX 변환·checkpoint mount·capacity 명령은 [deployment/README.md](../deployment/README.md#학습-서버에서-export)에만 유지한다. 기존 BEVFusion 학습 checkpoint와 신규 A/B/C deployment ABI는 별개다.

## 8. 검증 범위와 제한

2026-09-07 기록: 기존 extension 12개 빌드·신규 registry 등록, 기존 3-camera/6-camera 및 신규 config resolve, 신규 DynamicBEVFusion의 실제 nuScenes-mini 1-step 학습을 확인했다.

| 신규 mini smoke 기록 | 값 |
|---|---|
| 입력 | 6 views / 32,250 points / 15 boxes |
| 모델 파라미터 | 44,521,340 |
| 실행 | forward / backward / optimizer step |
| gradient parameter 수 | camera 190 / LiDAR 194 / fusion 54 / head 42 |
| peak CUDA allocated | 4.612 GiB |

이번 문서 정리에서는 세 config의 기본값과 train override resolve, 평가 DictAction의 null 파싱, smoke/test CLI help 및 명령의 코드 경로를 CPU에서 확인했다. 데이터 재생성·GPU 학습·resume·평가를 새로 실행하지 않았다. 기존 smoke 기록은 이후 변경된 학습 로직까지 검증한 결과가 아니다.

### 신규 모델 nuScenes full 학습 결과 (2026-09-10)

| 항목 | 값 |
|---|---|
| 데이터 | nuScenes v1.0-trainval, train 28,130 / val 6,019 |
| 환경 | 8 × RTX A6000 48GB, DDP, global batch 32, 20 epoch, 약 23시간 |
| 정밀도 | FP32 (`--fp16 None`). FP16은 cyclic LR 약 9.2e-4 도달 시 NaN → Hungarian assigner 실패로 중단 |
| 구성 | `dsvt_dgf_dal_widthformer_0p3.yaml`, single-sweep, GT-Aug(ObjectPaste) 미사용, ResNet34 ImageNet 초기화 |
| 실행 | `torchrun --nproc_per_node=8 tools/train_torchrun.py <config> --find_unused_parameters True --fp16 None` |
| 최고 성능 | epoch 19: mAP 0.5281 / NDS 0.5132 (mATE 0.318 / mASE 0.257 / mAOE 0.643 / mAVE 1.132 / mAAE 0.291) |
| 최종 epoch | epoch 20: mAP 0.5258 / NDS 0.5129 |
| 선택 checkpoint | `runs/full_fp32_fix/best_object/nds_epoch_19.pth` |
| 필요 코드 | commit `000230e` 이상 (DAL heatmap `[Y,X]` 타깃 수정 + ResNet34 `init_cfg`) |

mAVE 1.132는 단일 sweep에서 속도가 관측 불가능한 구조적 결과이며 NDS 속도 항에 0으로 반영된다. 클래스별로는 car 0.806 / pedestrian 0.767이 높고 construction_vehicle 0.165 / bicycle 0.255 / trailer 0.385가 낮다. 이 세 클래스는 GT-Aug 효과가 큰 희소 클래스다. 후속 우선순위는 GT-Aug 추가 재학습, 학습 weight 기준 FP16 engine 재검증, 10-sweep 대조 실험 순이다.

현재 **기존 모델의 전체 학습, 신규 학습 weight의 TensorRT 수치 일치·정확도, resume 및 위 MPI 평가 명령의 end-to-end 완료를 확인한 상태는 아니다**. Docker image build/컨테이너 학습 역시 미실행이며 환경 검사 범위는 [Docker 문서](../docker/README.md#소스-변경과-검증-상태)를 따른다. 이전 Thor inference 수치는 학습 검증이 아니며 100만 point 학습·35 ms 성능을 보장하지 않는다.
