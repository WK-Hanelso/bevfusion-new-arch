# EXEC — spconv v2 선택 백엔드 구현 결과

## 결과

`SparseEncoder`에 opt-in spconv v2 경로를 추가했다. 아무 설정이 없을 때는
기존 bundled spconv v1 경로를 사용하므로 기본 동작은 바뀌지 않는다.

- 환경 변수: `BEVFUSION_SPCONV=v2`
- config 인자: `spconv_backend: v2`
- config 인자가 있으면 환경 변수보다 우선한다.
- 허용 값은 `legacy`, `v2`이며 기본값은 `legacy`다.

구현 파일:

- `mmdet3d/ops/spconv_compat.py`: 백엔드 선택, 지연 import, state dict 키와
  weight layout 변환, 전체 checkpoint 재귀 로드용 pre-hook
- `mmdet3d/ops/sparse_block.py`: 선택 백엔드의 convolution 및
  `SparseSequential` 생성, v2용 residual basic block
- `mmdet3d/models/backbones/sparse_encoder.py`: 인스턴스별 백엔드 선택과
  `SparseConvTensor`/`SparseSequential` 연결
- `tests/test_spconv2_equiv.py`: CPU 호환 테스트와 GPU 동치 테스트

## checkpoint 호환

모듈 계층과 state dict 키는 두 백엔드에서 같게 유지했다. convolution
weight는 다음과 같이 변환한다.

```text
legacy: (kD, kH, kW, in, out)
v2:     (out, kD, kH, kW, in)
```

v2 convolution에는 load pre-hook을 등록했다. 전체 detector checkpoint를
재귀적으로 로드할 때 legacy shape가 v2 목표 shape와 일치하도록 변환될 수
있으면 자동 변환한다. shape가 우연히 동일해 자동 판별할 수 없는 모델은
아래 명시적 변환 함수를 사용한다.

```python
from mmdet3d.ops.spconv_compat import convert_spconv_state_dict

converted = convert_spconv_state_dict(
    legacy_state_dict,
    v2_model.state_dict(),
    source_backend="legacy",
    target_backend="v2",
)
v2_model.load_state_dict(converted, strict=True)
```

## 서버 설치

2026-09-19 확인 결과 PyPI에는 CPython 3.10/Linux용
`spconv-cu126==2.3.8` wheel이 있고 `spconv-cu128` 프로젝트는 없다.
`spconv-cu126` wheel은 manylinux_2_28이므로 서버 glibc가 2.28 이상이어야
한다. upstream은 spconv 2.x가 PyTorch binary에 의존하지 않으며 CUDA 11
이후 같은 major 계열의 minor-version 차이를 허용한다고 설명한다. 따라서
torch 2.7.1+cu128 환경의 우선 후보는 cu126 wheel이지만, 이 CPU 호스트에서
그 조합을 실행 확인한 것은 아니다. 실제 학습 서버에서 아래 GPU 동치
테스트를 통과시킨 뒤 사용한다.

기존 pip spconv/cumm 계열이 있다면 upstream 지침대로 충돌 패키지를 먼저
확인하고 제거한다.

```bash
python -m pip list | grep -E '^(spconv|cumm)'
python -m pip uninstall -y spconv spconv-cu124 spconv-cu126 cumm cumm-cu124 cumm-cu126
python -m pip install --upgrade pip
python -m pip install spconv-cu126==2.3.8
python -c "import torch, spconv; import spconv.pytorch; print(torch.__version__, spconv.__version__)"
```

cu126 wheel을 사용할 수 없는 OS/Python 조합에서는 `spconv-cu124`도 같은
CUDA 12 major 계열 후보이나 동일한 GPU 검증이 필요하다. prebuilt wheel과
driver 조합이 맞지 않으면 CUDA 12.8 환경에서 cumm/spconv를 source build하는
것이 대안이다.

사용 예:

```bash
BEVFUSION_SPCONV=v2 python train.py ...
BEVFUSION_SPCONV=v2 python -m pytest -q tests/test_spconv2_equiv.py
```

또는 SparseEncoder config에 다음을 추가한다.

```yaml
spconv_backend: v2
```

## 검증

현재 호스트 조건:

- CPU host (`torch.cuda.is_available() == False`)
- `spconv` pip package 없음
- 일반 `mmdet3d`/`mmcv` runtime import 불가

실행 결과:

```text
$ python -m pytest -q tests/test_spconv2_equiv.py
....s                                                                    [100%]
4 passed, 1 skipped
```

통과 항목:

1. 설정 없음에서 legacy 기본값 및 config 우선순위
2. legacy ↔ v2 weight layout round trip
3. state dict 키 보존과 shape 기반 변환
4. 재귀 load pre-hook의 legacy weight 수용

skip 항목:

- 동일 legacy 가중치와 동일 voxel 입력을 사용한 두 `SparseEncoder`의 fp32
  GPU 출력 비교 (`max_abs_diff <= 1e-3`): CUDA 또는 `spconv.pytorch`가
  없으면 skip한다.

구문 검증도 통과했다.

```bash
python -m py_compile \
  mmdet3d/ops/spconv_compat.py \
  mmdet3d/ops/sparse_block.py \
  mmdet3d/models/backbones/sparse_encoder.py \
  tests/test_spconv2_equiv.py
```

## 남은 서버 확인

GPU 동치 테스트는 이 호스트에서 실행할 수 없었다. 학습 서버에서 위 pytest
명령을 먼저 실행하고, 이어서 실제 checkpoint `strict=True` 로드와 짧은
forward/backward smoke test를 수행해야 한다. 이 확인 전에는
`BEVFUSION_SPCONV` 기본값을 v2로 바꾸지 않는다.

참고:

- https://pypi.org/project/spconv-cu126/
- https://github.com/traveller59/spconv
- https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html
