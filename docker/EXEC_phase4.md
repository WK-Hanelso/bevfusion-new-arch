# Phase 4 실행 기록 — B200 학습 환경

기준: `docker/SPEC_phase4_b200.md` §2 D1~D3. 이 문서는 로컬에서 수행한
Round 1 변경과, B200 오케스트레이터가 다음 라운드에 수행할 검증을 구분한다.

## Round 1 구현

### D1: torch 2.7 + CUDA 12.8 stack

- 실제 B200 접속 대상은 이미 container이므로 `b200_conda_env.yml`과
  `setup_b200_conda.sh`를 기본 설치 경로로 추가했다. script는 GPU/nvcc를 먼저
  출력하고 system CUDA 12.8.x/12.9.x를 재사용하거나, `nvidia` channel에서
  `cuda-toolkit=12.8`을 설치한다. CUDA 13 nvcc는 torch cu128 extension build의
  major-version mismatch를 피하기 위해 자동 재사용하지 않는다.
- conda 경로는 B200/Hopper에 필요한 `TORCH_CUDA_ARCH_LIST=9.0;10.0` 및
  `BEVFUSION_CUDA_ARCHS=90;100`으로 source extension을 빌드한다. source package는
  재실행 시 force-reinstall하여 다른 torch ABI로 만든 동일 버전 binary를 남기지
  않는다.
- `Dockerfile.cu128`: CUDA 12.8.1 devel / Ubuntu 22.04 / Python 3.10을
  사용하는 참조·재현 경로다. 공식 cu128 `torch 2.7.1`과 대응
  `torchvision 0.22.1`을 먼저 설치한다.
- `mmcv-full 1.7.2`는 `MMCV_WITH_OPS=1`, `FORCE_CUDA=1`,
  `TORCH_CUDA_ARCH_LIST=8.6;8.9;9.0;10.0`으로 소스 빌드한다. 같은 torch/CUDA
  header를 쓰도록 build isolation을 끈다.
- `torch-scatter 2.1.2`도 wheel fallback 없이 같은 이미지에서 소스
  빌드한다. `mmdet 2.28.2`, Torchpack 0.3.1, nuScenes devkit 1.1.11과
  Python 3.10용 최신 Numba 계열(Numba 0.61.2 / llvmlite 0.44.0)을
  `requirements-cu128.txt`에 명시했다.
- nuScenes devkit 1.1.11의 실제 package metadata가 요구하는
  `matplotlib<3.6`, `Shapely<2`는 유지하고, 그 밖의 일반 runtime package는
  불필요하게 2021년 버전으로 고정하지 않고 호환 범위를 열었다.

근거:

- PyTorch 공식 2.7 release는 Blackwell 지원과 CUDA 12.8 wheel을 명시한다.
  2.7.1 공식 설치 조합은 `torch 2.7.1` / `torchvision 0.22.1` / cu128이다.
  - https://pytorch.org/blog/pytorch-2-7/
  - https://pytorch.org/get-started/previous-versions/
- mmcv 1.7.2 `setup.py`는 torch 1.12 초과에서 C++17을 선택하고,
  `MMCV_WITH_OPS=1` 및 `FORCE_CUDA=1` source build 경로를 제공한다.
  - https://github.com/open-mmlab/mmcv/blob/v1.7.2/setup.py
- mmdet 2.28.2 자체 version guard는 mmcv 1.3.17~1.8.0을 허용하므로
  mmcv-full 1.7.2가 범위 안이다.
  - https://github.com/open-mmlab/mmdetection/blob/v2.28.2/mmdet/__init__.py

### D2: 저장소 CUDA/Python 호환

- `setup.py`는 CUDA 11.3에서는 기존 sm_70/75/80/86을 그대로 선택하고,
  toolkit 지원 시 sm_89(CUDA 11.8), sm_90(CUDA 12.0), sm_100(CUDA 12.8)을
  단계적으로 추가한다. `BEVFUSION_CUDA_ARCHS`를 지정하면 명시 목록으로
  override한다. 이 조건부 기본값이 기존 cu113 Docker build가 새 architecture를
  알지 못하는 nvcc에서 실패하는 것을 막는다.
- C/CUDA 전체 grep 결과 `THC/`, `THCState`, `THCuda*`, `AT_CHECK` 직접 사용은
  0건이었다. 대신 torch 2.7에서 실제로 사라진 private helper
  `at::cuda::ATenCeilDiv`가 voxel op 두 파일에 11건 있었다.
  `voxelization_cuda.cu`와 `scatter_points_cuda.cu`에서 private
  `CUDAApplyUtils.cuh` include와 해당 호출만 동등한 local integer ceil-div로
  치환했다. kernel/block 계산 외 동작은 바꾸지 않았다.
- `AT_CUDA_CHECK`는 torch 2.7의 `ATen/cuda/Exceptions.h`에도
  `C10_CUDA_CHECK` alias로 남아 있어 치환하지 않았다. `AT_ASSERTM`도 2.7에서
  deprecated alias일 뿐 제거되지 않았으므로 모델/오류 의미를 바꾸지 않았다.
- NumPy 1.24에서 만료·제거된 `np.bool`, `np.long` 사용을 저장소 전체에서 찾아
  각각 `np.bool_`, label contract에 맞는 `np.int64`로 바꿨다. 두 이름은
  NumPy 1.22에도 있어 cu113과 양쪽 호환이다.
- `torch._six` 사용은 0건이었다. `mmcv.runner` import 전체와 mmdet import
  전체를 1.7.2/2.28.2 export와 대조했고 현재 사용 symbol이 모두 남아 있음을
  확인했다. `mmdet3d/core/voxel/builder.py`만 `import mmcv` 뒤 우연한
  submodule side effect에 기대지 않도록 1.4/1.7 모두 제공하는
  `from mmcv.runner import obj_from_dict`로 명시화했다.
- `tools/train_torchrun.py`의 `distutils.version`은 Python 3.10에서 deprecated지만
  제거되는 것은 Python 3.12다. 목표 Python 3.10과 기존 3.8 모두 제공하며,
  torch 1.10 tensorboard lazy import 호환용이므로 이번에는 유지했다.

API 근거:

- torch 2.7.1 `CUDAApplyUtils.cuh`에는 `ATenCeilDiv`가 없고 공개
  `ATen/ceil_div.h`를 사용한다. 저장소는 가장 작은 양쪽 호환 치환을 위해
  local helper를 사용했다.
  - https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/cuda/CUDAApplyUtils.cuh
- torch 2.7.1 CUDA exception header에서 `AT_CUDA_CHECK`는
  `C10_CUDA_CHECK`로 정의되어 있다.
  - https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/cuda/Exceptions.h
- NumPy는 builtin alias를 1.20에서 deprecated했고 1.24에서 deprecation을
  만료시켜 오류로 전환했다.
  - https://numpy.org/doc/1.24/release/1.20.0-notes.html
  - https://numpy.org/doc/1.24/release/1.24.0-notes.html
- Python `distutils`는 3.10에서 deprecated, 3.12에서 제거되었다.
  - https://docs.python.org/3.12/library/distutils.html
- mmcv 1.7.2 runner는 BaseModule, fp16 decorators, optimizer/runner,
  checkpoint/distributed symbol과 `obj_from_dict`를 계속 export한다.
  - https://github.com/open-mmlab/mmcv/blob/v1.7.2/mmcv/runner/__init__.py

### D3: 환경 검사와 문서

- `check_environment.py`에 `cu113`과 `cu128` profile을 두고 `--stack auto`,
  `--stack cu113`, `--stack cu128`을 지원한다. 기존 strict contract를 보존하면서
  cu128에서는 Python 3.10, torch CUDA 12.8, 핵심 package와 torch wheel의
  `sm_100` 포함 여부를 검사한다.
- `docker/README.md`에 B200 build/run, host 사전 확인, architecture override,
  strict CUDA 검사와 후속 D4 순서를 추가했다. 실제 B200용 conda 설치와 8장
  `torchrun` 분할 예시를 기본 경로로 두고 Docker는 재현 경로로 구분했다.
- `b200_bootstrap_conda.sh`는 conda setup, strict CUDA 검사, 16개 smoke를
  순서대로 실행하고 checkpoint가 준비돼 있으면 full-model load까지 실행한다.

## 로컬 검증

컴파일/CUDA 실행은 이 host에서 수행하지 않는다. 아래 결과는 Round 1 종료 때
갱신한다.

```text
python -m py_compile <변경 Python 파일>: PASS
bash -n docker/setup_b200_conda.sh docker/b200_bootstrap_conda.sh: PASS
b200_conda_env.yml yaml.safe_load: PASS
PYTHONPATH=. python -m pytest -q tools/ablation/tests tools/dsvt_pretrained/tests deployment/runtime/tests: PASS (105 passed, 3 warnings, 12.42s)
legacy/import grep 재검사: PASS (제품 코드 잔여 0건)
git staging: PENDING
```

경고 3건은 기존 `scatter_reduce_` beta 경고 1건과 공식 DSVT reference의
`np.int` 경고 2건이다. 테스트의 `np.int = int` 1건은 변경 불가능한 공식 reference
module을 NumPy 신버전에서 실행하기 위한 의도적 shim이며 제품 코드 사용이 아니다.

## B200 다음 라운드 (D4)

1. `nvidia-smi`, `nvcc --version`, 실제 nuScenes full/mini 경로 확인.
2. `bash docker/setup_b200_conda.sh` 또는 inventory/smoke까지 묶은
   `bash docker/b200_bootstrap_conda.sh --data-root ...`.
3. `conda run -n bevfusion-b200 python docker/check_environment.py --strict --stack cu128 --cuda`.
4. `tools/ablation/smoke_all.sh` 16조합 mini 1-step.
5. `tools/dsvt_pretrained/check_full_model_load.py`.
6. `c4_full` mini 20 epoch로 loss 하강과 평가 pipeline 확인.
7. 고정 seed/동일 입력으로 cu113 forward와 비교(허용 오차 1e-3).

Round 1에서는 conda/Docker CUDA compiler 및 GPU 실행이 불가하므로 위 항목을
성공으로 기록하지 않는다. 특히 mmcv-full 1.7.2와 repository CUDA extension의
torch 2.7 실컴파일 결과는 다음 라운드의 오류 피드백 대상으로 남긴다.
