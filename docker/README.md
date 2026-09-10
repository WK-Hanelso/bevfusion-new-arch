# 학습용 Docker 환경

[저장소](../README.md) · [모델/config/학습](../new-arch.md) · [ONNX export](../deployment/README.md)

`docker/Dockerfile`은 기존 BEVFusion 3/6-camera와 신규 DynamicBEVFusion을 **같은 x86_64 GPU 학습 이미지**에서 실행한다. config와 `tools/train.py` 구조를 그대로 사용하며 ONNX export 의존성도 포함한다. Thor TensorRT 실행 환경은 [별도 배포 문서](../deployment/tensorrt/README.md)를 따른다.

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
cd /home/culee/workspace/bevfusion
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
cd /home/culee/workspace/bevfusion
mkdir -p runs .cache/bevfusion deployment/artifacts

docker run --rm -it --gpus all --shm-size=16g \
  --user "$(id -u):$(id -g)" \
  -v /home/culee/nuscenes_mini:/workspace/bevfusion/data/nuscenes:ro \
  -v "$PWD/runs:/workspace/bevfusion/runs" \
  -v "$PWD/.cache/bevfusion:/workspace/cache" \
  -v "$PWD/deployment/artifacts:/workspace/bevfusion/deployment/artifacts" \
  -w /workspace/bevfusion \
  bevfusion-train:cu113
```

`runs/`에 checkpoint/log가, `deployment/artifacts/`에 export 결과가 남는다. Cache는 Torch pretrained weights, Matplotlib, Numba용이다. 원본 데이터와 info PKL은 미리 준비해 read-only로 제공한다. 실제 실행 계정에 host의 결과/cache 디렉터리 쓰기 권한이 있어야 한다.

기본 dataset root는 `data/nuscenes/`다. `samples/`, `sweeps/`, `nuscenes_infos_train.pkl`, `nuscenes_infos_val.pkl`이 필요하며 기존 ObjectPaste config는 GT database도 필요하다. Raw mini 다운로드만으로 모든 PKL이 자동 생성되지는 않는다. 데이터 전처리는 이미지 빌드와 별개다.

## 3. 컨테이너 안에서 검사와 학습

```bash
# 버전·extension·config 및 작은 CUDA kernel 검사
python docker/check_environment.py --strict --cuda

# 신규 모델 실제 mini 1-step
python tools/smoke_train.py \
  configs/nuscenes/det/transfusion/secfpn/lidar/dsvt_dgf_dal_widthformer_0p3.yaml \
  --dataroot data/nuscenes --device 0

# 신규 모델 전체 학습
python -m torch.distributed.run --nproc_per_node=1 tools/train_torchrun.py \
  configs/nuscenes/det/transfusion/secfpn/lidar/dsvt_dgf_dal_widthformer_0p3.yaml \
  --run-dir runs/new-arch
```

기존 모델은 마지막 명령의 config만 다음 중 하나로 바꾸고 서로 다른 `--run-dir`를 지정한다.

| 선택 | config |
|---|---|
| 기존 전방 3-camera | `configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50/convfuser.yaml` |
| 기존 6-camera | `configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50/convfuser_6cam.yaml` |

기존 detection smoke에서 map-expansion 파일이 없으면 `smoke_train.py ... --object-only`를 사용할 수 있다. 이 옵션은 map label loader만 제외하므로 ObjectPaste의 GT database 요구까지 제거하지는 않는다.

멀티 GPU는 `--nproc_per_node`를 컨테이너에 노출된 GPU 수에 맞춘다. `train_torchrun.py`는 동일한 `tools/train.py`를 실행한다. OpenMPI를 사용하는 기존 `torchpack dist-run -np N python tools/train.py <config> --run-dir <dir>`도 가능하나, 위 명령은 root MPI 실행 옵션을 요구하지 않는 torchrun 경로를 기준으로 한다.

신규 PTH export는 같은 이미지에서 [deployment/README.md](../deployment/README.md)의 `export_all.py` 명령을 사용한다. TensorRT engine 생성은 해당 ONNX artifact를 Thor로 전달한 후 수행한다.

## 소스 변경과 검증 상태

소스를 변경하면 같은 `docker build` 명령으로 이미지를 갱신한다. `-v "$PWD:/workspace/bevfusion"`처럼 전체 root를 mount하면 이미지에서 컴파일한 `.so`가 host 소스로 가려지거나 host binary와 섞일 수 있다. 개발용 mount를 선택할 경우 컨테이너 안에서 extension을 다시 빌드해야 하며 host binary를 그대로 재사용하지 않는다.

2026-09-10 확인: 공식 GPU wheel 네 개와 Numba/llvmlite Python 3.8 wheel 제공, 학습 requirements pip dry-run, 기존 venv에서 extension 12개 import 및 세 모델 config resolve를 확인했다. Docker용 OpenCV/YAPF/protobuf를 별도 임시 overlay에 설치한 strict 검사도 통과했고 MMCV config formatting, TensorBoard, ONNX import를 확인했다. 기존 venv는 변경하지 않았다. 현재 서버에는 Docker CLI/daemon이 없어 **이 Dockerfile의 실제 image build와 컨테이너 학습은 아직 실행하지 않았다**. 이전 venv mini 학습 및 Thor inference 실측과 Docker 검증 상태를 구분한다.

공식 설치 자료: [PyTorch 이전 버전](https://pytorch.org/get-started/previous-versions/), [MMCV cu113/torch1.10.0 wheels](https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html), [torch-scatter cu113 wheels](https://data.pyg.org/whl/torch-1.10.0%2Bcu113.html). 실제 사용 파일 URL은 `requirements-cuda.txt`를 따른다.
