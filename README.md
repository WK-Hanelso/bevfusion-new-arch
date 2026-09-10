# BEVFusion — new-arch

팀의 기존 BEVFusion과 신규 DynamicBEVFusion을 같은 `configs/`, `mmdet3d/`, `tools/` 구조에서 선택해 학습하는 저장소다. 신규 모델은 WidthFormer camera branch, DSVT LiDAR branch, DepthGFusion과 DAL head로 구성된다.

- 저장소: [model-team/bevfusion](https://adtc.swm.ai/gitlab/model-team/bevfusion), 작업 branch: `new-arch`
- 팀 공유: [모델·빌드·실행 가이드](https://www.notion.so/3d0f51a0d215813a9f69fc27653475e0)
- 기존 Thor 실측을 inference 실행 구조의 검증 근거로 사용한다. 이번 정리는 재실측을 요구하지 않는다.

## 모델 선택

아래 YAML을 학습 진입점에 전달한다. 모델 종류와 camera 순서는 config가 결정하며, 임의 branch 조합이나 기존 3-camera 모델의 신규 6-camera TensorRT engine 사용을 의미하지 않는다.

| 구성 | config (`configs/nuscenes/det/transfusion/secfpn/` 기준) | 모델 |
|---|---|---|
| 팀 기존 전방 3-camera | `camera+lidar/resnet50/convfuser.yaml` | `BEVFusion` |
| 기존 모델의 nuScenes 6-camera | `camera+lidar/resnet50/convfuser_6cam.yaml` | `BEVFusion` |
| 신규 nuScenes 6-camera | `lidar/dsvt_dgf_dal_widthformer_0p3.yaml` | `DynamicBEVFusion` |

현재 `deployment/`의 PTH → ONNX → TensorRT → C++ 경로는 **신규 모델의 명시된 A/B/C 입출력 계약**을 대상으로 한다. 기존 BEVFusion 학습 config는 유지되지만, 기존 모델의 NVIDIA native C++ 배포 경로를 이 저장소의 `deployment/`가 제공하는 것은 아니다.

## 파일 구조와 문서

```text
configs/                 모델·camera·dataset·학습 설정
mmdet3d/                 기존 모델과 신규 모델 구현, dataset, 기존 CUDA ops
tools/                   train.py, train_torchrun.py, smoke_train.py, test.py
docker/                  학습 Dockerfile, 고정 requirements, 환경 검사
deployment/
  onnx/                  PTH → ONNX 및 LiDAR weights/manifest
  tensorrt/              A/B/C engine builder와 plugins/
  runtime/               모듈화된 C++ 추론, bundle launcher, tests/, evidence/
  artifacts/             생성물 전용, Git 제외
data/, runs/, build/     로컬 데이터·학습 결과·빌드 결과, Git 제외
```

| 문서 | 단일 책임 |
|---|---|
| [new-arch.md](new-arch.md) | 모델 선택, 아키텍처 그림, PyTorch 파일·입출력, 학습 방법 |
| [docker/README.md](docker/README.md) | 학습 이미지 빌드, 패키지 버전, 데이터/결과 mount, 컨테이너 학습 |
| [deployment/README.md](deployment/README.md) | PTH export, 생성 파일과 engine I/O 계약 |
| [deployment/tensorrt/README.md](deployment/tensorrt/README.md) | Thor Docker, plugin CMake, engine build 및 capacity |
| [deployment/runtime/README.md](deployment/runtime/README.md) | C++ 실행, latency·memory 옵션, 이전 실측 근거 |

## Docker 학습 환경

기존 BEVFusion과 신규 모델은 같은 학습 이미지를 사용한다. Repository root에서 빌드한다.

```bash
docker build --platform linux/amd64 -f docker/Dockerfile \
  --build-arg MAX_JOBS=4 -t bevfusion-train:cu113 .
```

이미지는 CUDA 11.3/Python 3.8/PyTorch 1.10.1의 x86_64 학습 환경이며 기존 CUDA ops 12개를 빌드한다. 실행·데이터 mount·환경 검사는 [docker/README.md](docker/README.md)를 따른다. 현재 작성본의 Docker image build는 아직 실행하지 않았으며 Thor TensorRT 이미지는 별도다.

## 기존 환경에서 시작

학습 서버의 준비된 venv는 `/home/culee/2608_bevfusion`, mini 데이터는 `/home/culee/nuscenes_mini`다. 다른 환경에서는 경로를 자신의 위치로 바꾼다.

```bash
cd /home/culee/workspace/bevfusion
source /home/culee/2608_bevfusion/bin/activate

python tools/smoke_train.py \
  configs/nuscenes/det/transfusion/secfpn/lidar/dsvt_dgf_dal_widthformer_0p3.yaml \
  --dataroot /home/culee/nuscenes_mini --device 0
```

`smoke_train.py`는 실제 1 batch의 forward/backward/optimizer step을 검사한다. 이전 신규 모델 mini smoke는 통과했다. 전체 학습 명령과 기존 config의 데이터 준비 조건은 [new-arch.md](new-arch.md#5-환경과-실행)를 따른다.

GPU 없는 bundle 계약 테스트:

```bash
python -m unittest discover -s deployment/runtime/tests -v
```

학습 checkpoint가 준비되면 [export](deployment/README.md#학습-서버에서-export) → [Thor engine build](deployment/tensorrt/README.md) → [C++ 실행](deployment/runtime/README.md) 순서로 진행한다. PTH/ONNX/engine/plugin 바이너리는 Git 소스와 분리해 전달하고, engine 배포에는 bundle/build manifest 및 동일 hash의 plugin 다섯 개를 포함한다.

## 현재 범위

- **검증 근거:** 신규 모델 nuScenes-mini 1-step 학습, 실제 PTH의 ONNX export, 이전 Thor의 FP16 A/B/C 실행·CUDA stream overlap·latency·통합 메모리 기록, 현재 bundle launcher CPU 테스트 18개.
- **이전 Thor 대표값:** 34,688 synthetic points에서 병렬 P50 26.4405 ms / P99 27.3075 ms(100회). 모델 연산 및 GPU DAL decode 기준이며 입력 전처리·H2D·출력 D2H는 제외한다. [원본 근거](deployment/runtime/evidence/thor_20260903_cuda_overlap_analysis.json).
- **별도 확정 사항:** 실제 5-LiDAR 통합 입력 분포, camera 구성, point feature 의미, pillar/set capacity, 실센서 입출력 연동, 학습 모델 정확도, 100만 point 설정의 성능·메모리. 100만 point는 고정값이 아니며 profile·capacity로 선택한다.

기존 실측은 random-init engine과 synthetic 입력의 구조 진단이다. 현재 학습 weight와 제품 센서 조합의 정확도·35 ms 성능을 측정했다는 뜻은 아니다. 기존 실행 근거와 현재 패키징 상태를 같은 문서에서 관리하며 버전별 문서를 추가하지 않는다.
