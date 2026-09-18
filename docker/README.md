# 학습용 Docker 환경

[저장소](../README.md) · [모델 아키텍처](../new-arch.md) · [학습·재개·평가](../tools/README.md) · [ONNX export](../deployment/README.md)

`docker/Dockerfile`은 기존 BEVFusion 3/6-camera와 신규 DynamicBEVFusion을 **같은 x86_64 GPU 학습 이미지**에서 실행한다. config와 `tools/train.py` 구조를 그대로 사용하며 ONNX export 의존성도 포함한다. Thor TensorRT 실행 환경은 [별도 배포 문서](../deployment/tensorrt/README.md)를 따른다.

## B200 (Blackwell, sm_100) 학습 이미지

실제 B200 접속 환경은 이미 GPU container이므로 nested Docker를 사용하지 않고
conda 환경을 만든다. `setup_b200_conda.sh`는 기존 system CUDA 12.8.x/12.9.x
`nvcc`를 우선 재사용하고, 없으면 `nvidia` channel의 `cuda-toolkit=12.8`을
`bevfusion-b200` 환경에 설치한다. 같은 명령을 재실행할 수 있으며 source extension은
현재 torch/CUDA에 맞게 다시 빌드한다.

```bash
cd /path/to/bevfusion
bash docker/setup_b200_conda.sh

# system CUDA 사용을 명시적으로 요구할 때(12.8.x/12.9.x가 아니면 즉시 실패)
bash docker/setup_b200_conda.sh --use-system-cuda

conda run -n bevfusion-b200 \
  python docker/check_environment.py --strict --stack cu128 --cuda
```

환경 구성, GPU 검사, nuScenes 경로 검사, 16개 1-step smoke를 한 번에 실행하려면
다음 bootstrap을 사용한다. full-model checkpoint가 있으면 load 검사도 이어서 한다.

```bash
bash docker/b200_bootstrap_conda.sh --data-root /path/to/nuscenes
```

8장 서버의 기본 운영은 서로 다른 GPU 4장씩 두 실험에 배정하는 방식이다. 각
shell/process에서 `CUDA_VISIBLE_DEVICES`를 분리하고 visible GPU 수와
`--nproc_per_node`를 맞춘다.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 conda run -n bevfusion-b200 \
  env PYTHONPATH=. torchrun --nproc_per_node=4 tools/train_torchrun.py \
  configs/nuscenes/det/ablation/dsvt0_wf0_gf0_dal0.yaml \
  --run-dir runs/exp-a

CUDA_VISIBLE_DEVICES=4,5,6,7 conda run -n bevfusion-b200 \
  env PYTHONPATH=. torchrun --nproc_per_node=4 tools/train_torchrun.py \
  configs/nuscenes/det/ablation/dsvt1_wf1_gf1_dal1.yaml \
  --run-dir runs/exp-b
```

`b200_conda_env.yml`은 Python/compiler/OpenMPI 기반을, `requirements-cu128.txt`는
Python runtime을 정의한다. setup script는 공식 cu128 torch wheel을 먼저 설치한
뒤 mmcv-full, torch-scatter, 저장소 ops를 `sm_90;sm_100`으로 소스 빌드한다.

### 재현용 Docker 이미지

`docker/Dockerfile.cu128`은 B200 stack의 참조·재현용 별도 이미지다. 기존 cu113
`Dockerfile`을 대체하거나 삭제하지 않는다. B200 접속 container 안에서는 이
이미지를 다시 build/run하지 않는다.

| 항목 | B200 구성 |
|---|---|
| Base | `nvidia/cuda:12.8.1-devel-ubuntu22.04`, Linux amd64 |
| Python / venv | Python 3.10, `/opt/bevfusion` |
| PyTorch / torchvision | 공식 cu128 wheel `2.7.1` / `0.22.1` |
| MMCV / MMDetection | `mmcv-full 1.7.2` CUDA ops 소스 빌드 / `mmdet 2.28.2` |
| DSVT scatter | `torch-scatter 2.1.2` cu128 소스 빌드 |
| CUDA extension target | `sm_86`, `sm_89`, `sm_90`, `sm_100` |

PyTorch 2.7 cu128은 Blackwell과 `sm_100`을 포함한다. MMCV 2.x/mmengine으로
이전하지 않고 저장소의 mmcv 1.x registry/API를 유지한다. `requirements-cu128.txt`는
Python runtime 의존성을 담고, CUDA ABI에 묶이는 torch/torchvision, mmcv-full,
torch-scatter는 Dockerfile에서 설치 순서와 source build를 강제한다.

독립 host에서 재현용 이미지를 쓸 경우 GPU, driver, Docker GPU passthrough와
데이터 위치를 먼저 확인한다. CUDA 12.8 Update 1에 대응하는 R570 계열 이상
driver를 권장한다.

```bash
nvidia-smi
docker version
docker info | sed -n '/Runtimes/,+2p'
test -d /path/to/nuscenes && find /path/to/nuscenes -maxdepth 1 -type d -print

docker build --platform linux/amd64 \
  -f docker/Dockerfile.cu128 \
  --build-arg MAX_JOBS=4 \
  -t bevfusion-train:cu128 .
```

메모리가 부족하면 `MAX_JOBS=2`로 낮춘다. 저장소 extension의 target은
`BEVFUSION_CUDA_ARCHS`로 덮어쓸 수 있으며 `86;89;90;100` 또는
`8.6;8.9;9.0;10.0` 형식을 받는다. 기존 CUDA 11.3 stack은 toolkit이 지원하는
기존 target(`70/75/80/86`)만 자동 선택한다.

```bash
docker run --rm -it --gpus all --shm-size=16g \
  -v /path/to/nuscenes:/workspace/bevfusion/data/nuscenes:ro \
  -v "$PWD/runs:/workspace/bevfusion/runs" \
  -v "$PWD/.cache/bevfusion:/workspace/cache" \
  -w /workspace/bevfusion \
  bevfusion-train:cu128

python docker/check_environment.py --strict --stack cu128 --cuda
```

환경 검사가 끝난 뒤 B200에서 SPEC의 D4 순서대로 16개 1-step smoke,
full-model load, mini 20 epoch, 기존 cu113 forward 수치 비교를 수행한다. 이번
로컬 라운드에서는 image/CUDA compilation과 GPU 실행을 하지 않았으며 수행 기록과
근거는 [EXEC_phase4.md](EXEC_phase4.md)에 둔다.

## 고정 환경과 파일

| 항목 | 구성 |
|---|---|
| Base | `nvidia/cuda:11.3.1-devel-ubuntu20.04`, Linux amd64 |
| Python / venv | Python 3.8 + 개발 header, `/opt/bevfusion` |
| PyTorch / torchvision | `1.10.1+cu113` / `0.11.2+cu113` |
| MMCV / MMDetection | `mmcv-full 1.4.0` / `mmdet 2.20.0`; `mmcv` 별도 설치 금지 |
| DSVT scatter | `torch-scatter 2.0.9`, torch 1.10 / cu113 wheel |
| 기존 numerical ops | NumPy 1.22.4, Numba 0.48.0, llvmlite 0.31.0, SciPy 1.7.3 |
| 학습 / dataset | Torchpack 0.3.1, nuScenes-devkit 1.1.11, OpenMPI + mpi4py 3.1.6 |
| Export | ONNX 1.14.1 |
| 호환성 고정 | OpenCV 4.8.1.78, YAPF 0.32.0, protobuf 3.20.3 |
| Build tools | GCC/G++, ninja, pip 24.0, setuptools 59.5.0, wheel 0.38.4 |

- `Dockerfile`: OS/Python 환경 → GPU wheels → 학습 dependencies → 소스 복사 → 기존 CUDA extension 12개 빌드 → 환경 검사.
- `requirements-cuda.txt`: 공식 Python 3.8/x86_64 wheel URL. CPU wheel이나 다른 CUDA build로 자동 대체하지 않는다.
- `requirements-training.txt`: 학습·nuScenes 평가·신규 ONNX export의 주요 버전 고정.
- `check_environment.py`: package version, extension import, registry, 세 config resolve 검사. `--cuda`는 작은 GPU scatter/voxelization 검사 추가.
- 저장소 root `.dockerignore`: 데이터, PTH/ONNX/engine, host에서 컴파일한 `.so`, build/runs/cache를 image context에서 제외.

기존 venv의 핵심 모델/수치 라이브러리 버전을 기준으로 했다. OpenCV/YAPF/protobuf는 새 이미지에서 각각 위 호환 버전으로 고정하며 기존 venv를 변경하지 않는다. YAPF 고정은 MMCV의 `FormatCode(verify=...)` 호출을 보존한다. OS apt package와 모든 간접 Python dependency까지 hash로 잠근 완전한 lockfile은 아니다.

## 1. 이미지 빌드

Host prerequisite: x86_64 Linux, Docker Engine, 실행 시 GPU를 노출할 NVIDIA driver/Container Toolkit. CUDA compiler와 Python 개발 header는 이미지 안에 설치한다. 기존 `setup.py`의 CUDA target은 `sm_70/75/80/86`; Thor `sm_110`용 학습 이미지가 아니다.

반드시 **repository root를 build context**로 사용한다.

```bash
cd bevfusion  # 저장소 루트
docker build --platform linux/amd64 \
  -f docker/Dockerfile \
  --build-arg MAX_JOBS=4 \
  -t bevfusion-train:cu113 .
```

빌드 시 GPU가 보이지 않아도 `FORCE_CUDA=1`로 기존 CUDA ops를 컴파일한다. CPU-only extension을 성공한 이미지로 남기지 않는다. 마지막에 `pip check`와 `python docker/check_environment.py --strict`가 실패하면 image build도 실패한다. RAM 부족으로 compiler가 종료되면 `--build-arg MAX_JOBS=2`로 병렬 compilation 수를 낮춘다.

인터넷 연결은 apt/package wheel 다운로드에 필요하다. 이미지 빌드 과정에서는 nuScenes 데이터, 학습 checkpoint, backbone pretrained weights를 다운로드하지 않는다. 실행 시 pretrained backbone을 쓰는 config는 최초 학습 때 네트워크 또는 준비된 torch cache가 필요하다.

## 2. 데이터·결과·cache mount

다음 예는 현재 학습 서버의 mini 경로다. 다른 host에서는 source 경로만 변경한다. Docker에는 소스와 그 소스용 `.so`가 이미 들어 있으므로 기본 사용에서는 repository root 전체를 bind mount하지 않는다.

```bash
cd bevfusion  # 저장소 루트
mkdir -p runs .cache/bevfusion deployment/artifacts

docker run --rm -it --gpus all --shm-size=16g \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/data/nuscenes_mini:/workspace/bevfusion/data/nuscenes:ro" \
  -v "$PWD/runs:/workspace/bevfusion/runs" \
  -v "$PWD/.cache/bevfusion:/workspace/cache" \
  -v "$PWD/deployment/artifacts:/workspace/bevfusion/deployment/artifacts" \
  -w /workspace/bevfusion \
  bevfusion-train:cu113
```

`runs/`에 checkpoint/log가, `deployment/artifacts/`에 export 결과가 남는다. Cache는 Torch pretrained weights, Matplotlib, Numba용이다. 원본 데이터와 info PKL은 미리 준비해 read-only로 제공한다. 실제 실행 계정에 host의 결과/cache 디렉터리 쓰기 권한이 있어야 한다.

기본 dataset root는 `data/nuscenes/`다. 데이터 구성·PKL/GT database 준비·mini/full 구분은 [학습 가이드](../tools/README.md#2-nuscenes-mini--full-데이터-준비)를 따른다. 전처리는 이미지 빌드와 별개이며 데이터 생성 단계에만 쓰기 가능한 mount를 사용한다.

## 3. 컨테이너 환경 검사

```bash
python docker/check_environment.py --strict --cuda
```

버전·extension·세 config와 작은 CUDA scatter/voxelization 연산을 검사한다. Dataset 학습이나 정확도 검사가 아니다. 검사 후 [학습·재개·평가 가이드](../tools/README.md)로 진행한다. 모델별 학습 명령과 MPI 평가의 전제조건은 그 문서에서만 관리한다.

신규 PTH export는 같은 이미지에서 [export 환경·checkpoint 경로](../deployment/README.md#학습-서버에서-export)의 `export_all.py` 명령을 사용한다. 컨테이너의 repository root는 `/workspace/bevfusion`이며 host venv의 `cd`/`source` 명령을 사용하지 않는다. 기본 mount의 학습 checkpoint는 `/workspace/bevfusion/runs/`에서 보이고, 외부 PTH는 위 문서의 read-only checkpoint mount를 추가해야 한다. TensorRT engine 생성은 해당 ONNX artifact를 Thor로 전달한 후 수행한다. Thor 이미지 확보·전달 방법과 재구성 한계는 [TensorRT 환경 문서](../deployment/tensorrt/README.md#thor-docker-prerequisite)를 따른다.

## 소스 변경과 검증 상태

소스를 변경하면 같은 `docker build` 명령으로 이미지를 갱신한다. `-v "$PWD:/workspace/bevfusion"`처럼 전체 root를 mount하면 이미지에서 컴파일한 `.so`가 host 소스로 가려지거나 host binary와 섞일 수 있다. 개발용 mount를 선택할 경우 컨테이너 안에서 extension을 다시 빌드해야 하며 host binary를 그대로 재사용하지 않는다.

2026-09-10 확인: 공식 GPU wheel 네 개와 Numba/llvmlite Python 3.8 wheel 제공, 학습 requirements pip dry-run, 기존 venv에서 extension 12개 import 및 세 모델 config resolve를 확인했다. Docker용 OpenCV/YAPF/protobuf를 별도 임시 overlay에 설치한 strict 검사도 통과했고 MMCV config formatting, TensorBoard, ONNX import를 확인했다. 기존 venv는 변경하지 않았다. 현재 서버에는 Docker CLI/daemon이 없어 **이 Dockerfile의 실제 image build와 컨테이너 학습은 아직 실행하지 않았다**. 이전 venv mini 학습 및 Thor inference 실측과 Docker 검증 상태를 구분한다.

공식 설치 자료: [PyTorch 이전 버전](https://pytorch.org/get-started/previous-versions/), [MMCV cu113/torch1.10.0 wheels](https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html), [torch-scatter cu113 wheels](https://data.pyg.org/whl/torch-1.10.0%2Bcu113.html). 실제 사용 파일 URL은 `requirements-cuda.txt`를 따른다.
