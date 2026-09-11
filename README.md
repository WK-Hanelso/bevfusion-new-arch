# BEVFusion — new-arch

팀의 기존 BEVFusion과 신규 DynamicBEVFusion을 같은 `configs/`, `mmdet3d/`, `tools/` 구조에서 선택해 학습하는 저장소다. 신규 모델은 WidthFormer camera branch, DSVT LiDAR branch, DepthGFusion과 DAL head로 구성된다.

- 저장소: `model-team/bevfusion`, 작업 branch: `new-arch`
- 정리 문서: [BEVFusion new-arch | 아키텍처·검증 결과·적용 가이드](https://app.notion.com/p/3d0f51a0d215813a9f69fc27653475e0) · [학습 구조 및 학습 결과](https://app.notion.com/p/3d7f51a0d21581a6b6a2ff5785c0cb68)
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
| [new-arch.md](new-arch.md) | 모델 선택, 아키텍처 그림, PyTorch 파일·입출력 |
| [docker/README.md](docker/README.md) | 학습 이미지 빌드, 패키지 버전, 데이터/결과 mount, 환경 검사 |
| [tools/README.md](tools/README.md) | 데이터 준비, smoke, 단일·멀티 GPU 학습, 재개·평가, checkpoint 전달 |
| [deployment/README.md](deployment/README.md) | PTH export, 생성 파일과 engine I/O 계약 |
| [deployment/tensorrt/README.md](deployment/tensorrt/README.md) | Thor Docker, plugin CMake, engine build 및 capacity |
| [deployment/runtime/README.md](deployment/runtime/README.md) | C++ 실행, latency·memory 옵션, 이전 실측 근거 |

## 환경 → 학습 → 배포

1. [Docker 환경](docker/README.md)에서 이미지 빌드·데이터/cache mount·환경 검사를 수행한다. 현재 Docker image build/컨테이너 학습은 미검증이며 Thor 이미지는 별도다.
2. [학습 가이드](tools/README.md)에서 host venv 또는 컨테이너 경로를 선택하고 데이터 준비 → smoke → 전체 학습 → 재개·평가를 진행한다.
3. 선택한 checkpoint와 config를 [ONNX export](deployment/README.md)에 전달한다. 명령은 각 담당 문서에서 관리한다.

GPU 없는 bundle 계약 테스트:

```bash
python -m unittest discover -s deployment/runtime/tests -v
```

학습 checkpoint가 준비되면 [export](deployment/README.md#학습-서버에서-export) → [Thor engine build](deployment/tensorrt/README.md) → [C++ 실행](deployment/runtime/README.md) 순서로 진행한다. PTH/ONNX/engine/plugin 바이너리는 Git 소스와 분리해 전달하고, engine 배포에는 bundle/build manifest 및 동일 hash의 plugin 다섯 개를 포함한다.

## 현재 범위

- **검증 근거:** 신규 모델 nuScenes full 학습·평가(아래), 실제 PTH의 ONNX export, 이전 Thor의 FP16 A/B/C 실행·CUDA stream overlap·latency·통합 메모리 기록, 현재 bundle launcher CPU 테스트 18개.
- **신규 모델 학습 결과 (2026-09-10):** nuScenes v1.0-trainval, 8×A6000, FP32, single-sweep, GT-Aug 미사용, 20 epoch. 최고 epoch 19 **mAP 0.5281 / NDS 0.5132**, 최종 epoch 20 mAP 0.5258 / NDS 0.5129. mAVE 1.132로 NDS 속도 항 0(단일 sweep의 구조적 결과). 재현 조건과 발견한 결함은 [학습 가이드의 검증 범위](tools/README.md#8-검증-범위와-제한) 참조.
- **이전 Thor 대표값:** 34,688 synthetic points에서 병렬 P50 26.4405 ms / P99 27.3075 ms(100회). 모델 연산 및 GPU DAL decode 기준이며 입력 전처리·H2D·출력 D2H는 제외한다. [원본 근거](deployment/runtime/evidence/thor_20260903_cuda_overlap_analysis.json).
- **별도 확정 사항:** 실제 5-LiDAR 통합 입력 분포, camera 구성, point feature 의미, pillar/set capacity, 실센서 입출력 연동, 학습 weight의 TensorRT 수치 일치·정확도·latency, 100만 point 설정의 성능·메모리. 100만 point는 고정값이 아니며 profile·capacity로 선택한다.

기존 실측은 random-init engine과 synthetic 입력의 구조 진단이다. 현재 학습 weight와 제품 센서 조합의 정확도·35 ms 성능을 측정했다는 뜻은 아니다. 기존 실행 근거와 현재 패키징 상태를 같은 문서에서 관리하며 버전별 문서를 추가하지 않는다.
