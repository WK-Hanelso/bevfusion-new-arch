# SPEC Phase 4 — B200(Blackwell, sm_100)용 학습 환경: torch 2.x + CUDA 12.8 스택 포팅

사용자 결정(2026-09-18): "B200 있는데 그걸로 부탁해." 본 학습(nuScenes full, ablation 16조합)을 B200에서 돌린다. 브랜치 `orin-trt85-plugins` 위에서 작업(별도 브랜치 `b200-env`로 분기).

## 1. 사실
- 현재 학습 이미지 `docker/Dockerfile` = CUDA 11.3.1 / torch 1.10.1+cu113 / mmcv-full 1.4.0(wheel) / mmdet 2.20.0 / torch-scatter 2.0.9 / Python 3.8 / Torchpack 0.3.1. **sm_100 불가**(CUDA ≥12.8 필요; torch 1.10 wheel에 sm_100 없음).
- 저장소 CUDA 확장: `setup.py`가 `-gencode` sm_70/75/80/86 고정. `mmdet3d/ops/` 16개 ops 중 레거시 THC/AT_CHECK 계열 사용 파일 1개(구현 시 grep으로 특정).
- 코드베이스는 mmcv 1.x API(`mmcv.Config`, `mmcv.runner`, `build_conv_layer`, `mmdet 2.x` registry)에 의존. mmcv 2.x(mmengine) 전환은 범위 밖.
- B200 호스트 정보(드라이버, docker, nuScenes 데이터 위치)는 미확인 → §4의 첫 단계에서 확인.

## 2. 설계 결정
- **D1 스택**: Python 3.10, CUDA 12.8 devel 베이스(`nvidia/cuda:12.8.1-devel-ubuntu22.04`), **torch 2.7.x + cu128**(공식 wheel, sm_100 포함), torchvision 대응 버전, **mmcv-full 1.7.2를 소스 빌드**(`MMCV_WITH_OPS=1`, `TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0;10.0"`), **mmdet 2.28.2**(mmcv-full 1.7 호환), mmsegmentation 불필요 시 제외, torch-scatter 소스 빌드(cu128), numba/llvmlite 최신, Torchpack 0.3.1, mpi4py, nuscenes-devkit, 기타 `docker/requirements-training.txt` 항목은 버전 상한 완화.
- **D2 저장소 확장**: `setup.py` gencode 목록에 `compute_89/sm_89`, `compute_90/sm_90`, `compute_100/sm_100` 추가(환경변수 `BEVFUSION_CUDA_ARCHS`로 override 가능). 레거시 API 파일은 `THC` → `ATen`/`c10::cuda` 등 최소 치환. mmcv 1.7 / mmdet 2.28 / torch 2.7에서 깨지는 import·API(`mmcv.runner`의 deprecated 항목, `torch._six`, `distutils`, `np.float` 등)를 수정하되 **cu113 이미지에서의 동작은 유지**(양쪽 호환 코드).
- **D3 파일**: `docker/Dockerfile.cu128`, `docker/requirements-cu128.txt`, `docker/README.md`에 B200 절, `docker/check_environment.py`가 두 스택 모두 통과.
- **D4 검증(B200에서, 오케스트레이터 실행)**: (1) 이미지 빌드, (2) `check_environment.py --strict`, (3) `tools/ablation/smoke_all.sh` 16개 1-step(mini), (4) `check_full_model_load.py`, (5) mini 20 epoch 짧은 학습 1회(c4_full)로 loss 하강·평가 파이프라인 확인, (6) 기존 cu113 이미지와 동일 입력 forward 수치 비교(허용 오차 1e-3, 랜덤 init 고정 seed).
- **D5 금지**: mmcv 2.x/mmengine 전환 금지. 모델 코드 동작 변경 금지. cu113 Dockerfile 삭제 금지.

## 3. codex 작업 범위(로컬, 컴파일 불가)
D1~D3 파일 작성 + D2 코드 수정 + 로컬 가능한 검증(python 문법, 기존 CPU 테스트 105개 통과). 빌드·실행은 B200 접속 후 오케스트레이터가 수행하고 오류를 다음 라운드로 전달. `docker/EXEC_phase4.md`에 기록.

## 5. 추가 결정 (2026-09-18, 사용자): conda 기반, B200 8장
- **B200 접속 환경 자체가 docker 컨테이너**이므로 그 안에서 docker를 다시 쓰지 않는다. **conda 환경**으로 구성한다. `docker/Dockerfile.cu128`은 참조·재현용으로 유지하되, 실제 설치 경로는 `docker/b200_conda_env.yml` + `docker/setup_b200_conda.sh`(멱등, 재실행 가능)이다.
- conda 구성: python 3.10, `nvidia` 채널 `cuda-toolkit=12.8`(nvcc 포함, mmcv·torch-scatter·저장소 ops 소스 빌드용), torch 2.7.x+cu128(pip 공식 index), torchvision 대응, mmcv-full 1.7.2 소스 빌드(`MMCV_WITH_OPS=1`, `TORCH_CUDA_ARCH_LIST="9.0;10.0"`), mmdet 2.28.2, torch-scatter 소스 빌드, Torchpack 0.3.1, mpi4py(pip), nuscenes-devkit, numba/llvmlite 최신, 저장소 `pip install -e . --no-deps --no-build-isolation`(`BEVFUSION_CUDA_ARCHS="9.0;10.0"`).
- 컨테이너에 nvcc가 이미 있을 수 있음(NGC 이미지 등) → 스크립트가 먼저 `nvcc --version`·`nvidia-smi`를 확인하고, 시스템 CUDA 12.8+가 있으면 conda cuda-toolkit 설치를 건너뛴다(플래그 `--use-system-cuda`).
- 검증 명령은 §2 D4와 동일하되 docker run 대신 `conda run -n bevfusion-b200`으로 실행. `docker/b200_bootstrap.sh`는 conda 버전으로 대체(`docker/b200_bootstrap_conda.sh`).
- 8장 활용: 학습은 `torchrun --nproc_per_node=<N>`으로 GPU 부분집합(`CUDA_VISIBLE_DEVICES`)에 배정. 동시 2개 실험 × 4 GPU가 기본.
