# Phase 1b 실행 기록 — 공식 DSVT 기본 구조와 배포 export 정합성

## 1. 시작 상태와 환경

- 시작 브랜치: `dsvt-pretrained-init`
- 시작 명령: `git status --short --branch`
- 시작 출력: `## dsvt-pretrained-init`, 미추적 파일은
  `tools/dsvt_pretrained/SPEC_phase1b.md` 하나였다. 기존 추적 파일 변경은 없었다.
- 기준 문서: `SPEC_phase1b.md` 전체를 먼저 읽고, 배경으로 `SPEC.md`와
  `EXEC.md`의 Phase 1 기록을 확인했다.
- Python/PyTorch: PyTorch `1.12.0+cu116`, `torch.cuda.is_available() == False`
- `onnx 1.13.1`, `PyYAML 6.0.3`, `g++` 사용 가능
- `mmcv`는 설치되어 있지 않으며 지침에 따라 패키지를 설치하지 않았다.

## 2. 구현

### D3 — export와 runtime 정합성

- `DSVTDeployWrapper`가 `backbone.official_layout`을 읽는다. 공식 구조에서는
  각 `forward_with_gather` 뒤에 해당 axis의 입력을 더하고
  `block.layer_norms[axis]`를 적용한다. legacy에서는 기존 연산을 유지한다.
- export 합성 topology의 gather도 공식 구조에서는
  `official_occurrence_gather`, legacy에서는 `last_occurrence_gather`를 사용한다.
- LiDAR artifact manifest와 TensorRT build manifest에 `official_layout`과
  `zero_feature_channels`를 기록하고 값의 타입, 범위, 중복을 검증한다.
- `DSVTBEVNeck` export wrapper는 전달받은 neck 객체를 그대로 보관하고
  `forward(dense_bev)`에서 단일 인자로 직접 호출하는 기존 ABI를 유지했다.
- bundle launcher는 hash 검증이 끝난 LiDAR build manifest 경로를 C++에
  `--lidar-manifest`로 전달한다. C++ runtime은 manifest의
  `zero_feature_channels`를 읽고 host point buffer에 적용한 뒤 H2D한다.
  plugin 소스는 수정하지 않았다.
- `point_features.{hpp,cpp}`를 TensorRT/CUDA 비의존 단위로 분리하고, 실제
  `[N,5]` buffer의 channel 4가 0이 되는 CPU C++ 테스트를 추가했다.

### D1 — 기본값

- `DSVTLidarEncoder`, `DSVTBackbone`, `DSVTBlock`, `SetAttention`의
  `official_layout` 기본값을 모두 `True`로 변경했다.
- encoder와 backbone에 선택값을 보관하며 production exporter는 encoder의
  값을 전달하고 두 값이 다르면 실패한다.
- `DynamicPillarVFE.zero_feature_channels` 기본값은 `[]` 그대로 유지했다.
- `DSVTBackbone.official_layout`을 명시적으로 보관해 export가 모델 설정을
  읽도록 했다.

### D2 — config

- 기본 `dsvt_dgf_dal_widthformer_0p3.yaml`에 `official_layout: true`와
  `zero_feature_channels: [4]`를 명시하고 공식 checkpoint 가이드 주석을
  추가했다.
- `dsvt_dgf_dal_widthformer_0p3_legacy.yaml`을 만들고
  `official_layout: false`, `zero_feature_channels: []` 및 ep19/Thor 재현 전용
  설명을 기록했다.
- 기존 `dsvt_dgf_dal_widthformer_0p3_dsvtpre.yaml`은 삭제했다.
- 두 config의 YAML 의미 비교 결과 차이는 아래 두 항목뿐이었다.

```text
model.encoders.lidar.official_layout: true -> false
model.encoders.lidar.zero_feature_channels: [4] -> []
```

### D4 — 테스트와 문서

- legacy snapshot/forward 테스트는 `official_layout=False`를 계속 명시한다.
- 기본 생성자 네 종류가 공식 구조를 선택하는 테스트를 추가했다.
- 공식/legacy export backbone wrapper, neck wrapper, manifest 계약 테스트를
  추가했다.
- `new-arch.md`, pretrained 가이드, deployment 및 runtime README를 기본 공식
  구조, legacy 절, H2D 전 `feature[4]` 0 처리 계약에 맞게 갱신했다.
- 기존 26.4 ms 계열 Thor 측정은 legacy 구조 기준임을 명시했다.

## 3. DoD 실행 결과

### 변환 checkpoint 재생성

```bash
python tools/dsvt_pretrained/convert_official_dsvt.py
```

```text
converted matched=336 missing=6 unexpected=113 elements=6232032
```

생성된 PTH/report는 기존 `.gitignore` 규칙에 따라 commit 대상이 아니다.

### T1, T2, T3 — Phase 테스트

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
pytest -q -s tools/dsvt_pretrained/tests/test_official_layout.py
```

최종 재실행 결과: **11 passed, 3 warnings, 8.65초**.

- 기존 Phase 1의 7개 검증은 기본값 변경에 맞춰 모두 통과했다.
- legacy encoder state snapshot과 main 원문 대비 forward는 bitwise 동일,
  `max_abs_diff=0`이었다.
- 변환 checkpoint는 공식 encoder에 로드됐고 missing은 adapter뿐,
  unexpected는 0이었다. 공식 이식 tensor elements는 6,232,032이다.
- 공식 checkpoint를 로드한 backbone에서 export wrapper와 직접 forward의
  `max_abs_diff=0`이었다.
- legacy backbone에서도 export wrapper와 직접 forward의
  `max_abs_diff=0`이었다.
- `DSVTBEVResNeck` export wrapper 출력은 직접 호출과 bitwise 동일했다.
- 기본 생성자와 LiDAR manifest의 공식/legacy 계약값을 확인했다.
- warning 3개는 PyTorch 1.12 `scatter_reduce_` beta 경고 1개와 공식 참조
  코드의 NumPy `np.int` deprecation 경고 2개다.

### T4 — deployment CPU 계약 테스트

```bash
PYTHONPATH=. pytest -q deployment
```

최종 재실행 결과: **20 passed, 1.38초**.

기존 launcher 계약 테스트와 함께 `official_layout`/
`zero_feature_channels` 검증, manifest 경로 전달, C++ host buffer channel 0
처리가 통과했다.

### 정적 검사와 config 검사

```bash
python -m py_compile \
  deployment/onnx/dsvt_backbone.py \
  deployment/onnx/export_lidar_trt_artifacts.py \
  deployment/onnx/model_contract.py \
  deployment/runtime/run_inference.py \
  deployment/tensorrt/lidar_builder.py \
  tools/dsvt_pretrained/tests/test_official_layout.py \
  deployment/runtime/tests/test_run_inference.py \
  deployment/runtime/tests/test_point_features.py

g++ -std=c++17 -Wall -Wextra -Wpedantic -Ideployment/runtime \
  -fsyntax-only deployment/runtime/options.cpp \
  deployment/runtime/point_features.cpp

git diff --check
```

모두 PASS했다. PyYAML 재귀 비교로 기본/legacy config가 두 LiDAR 옵션 외에는
동일하고, 삭제 대상 `_dsvtpre.yaml`이 존재하지 않음도 확인했다.

### T5 — 문서

- `new-arch.md`: 기본 공식 구조와 legacy 재현 절을 분리했다.
- `tools/dsvt_pretrained/README.md`: 기본 config를 사용하는 변환/학습 명령으로
  갱신했다.
- `deployment/README.md`: LiDAR 입력 계약과 legacy Thor 수치 범위를 명시했다.
- `deployment/runtime/README.md`: manifest를 읽는 H2D 전 channel 0 처리와 CPU
  테스트 범위를 기록했다.

## 4. 실행하지 못한 항목

### 실제 ONNX export

다음 명령은 import 단계에서 실패했다.

```bash
PYTHONPATH=. python deployment/onnx/export_lidar_trt_artifacts.py --help
```

```text
ModuleNotFoundError: No module named 'mmcv'
```

현재 환경은 CPU 전용이고 `torch.cuda.is_available() == False`이며 exporter도
CUDA를 요구한다. 패키지 설치 금지 지침에 따라 `mmcv`를 설치하지 않았다.
따라서 실제 ONNX 파일 생성과 ONNX checker 실행은 미검증이다. 대신 동일
wrapper의 PyTorch CPU 수치 동치와 Python compile을 검증했다.

### TensorRT/CUDA 및 Thor

TensorRT/CUDA headers와 GPU/Thor가 없는 환경이므로 전체 C++ runtime 빌드,
engine build, H2D/GPU inference, latency/정확도 측정은 실행하지 않았다.
TensorRT plugin 소스는 변경하지 않았다. manifest parsing 및 channel 0 함수는
독립 C++ CPU 테스트로 검증했지만, **공식 기본 구조의 Thor 실측은 미검증**이다.

## 5. git staging

아래 변경 파일을 명시해 `git add`를 실행했으나 `.git`이 read-only라 staging은
생성되지 않았다. 지침대로 실패를 무시했으며 commit은 실행하지 않았다.

```text
fatal: Unable to create '/home/hanelso/hanelso/bevfusion/.git/index.lock': Read-only file system
```

모든 변경은 working tree에 남아 있다.

## 6. 변경 파일 목록

- `configs/nuscenes/det/transfusion/secfpn/lidar/dsvt_dgf_dal_widthformer_0p3.yaml`
- `configs/nuscenes/det/transfusion/secfpn/lidar/dsvt_dgf_dal_widthformer_0p3_dsvtpre.yaml` (삭제)
- `configs/nuscenes/det/transfusion/secfpn/lidar/dsvt_dgf_dal_widthformer_0p3_legacy.yaml` (신규)
- `deployment/README.md`
- `deployment/onnx/dsvt_backbone.py`
- `deployment/onnx/export_lidar_trt_artifacts.py`
- `deployment/onnx/model_contract.py`
- `deployment/runtime/CMakeLists.txt`
- `deployment/runtime/README.md`
- `deployment/runtime/multisensor_pipeline.cpp`
- `deployment/runtime/options.cpp`
- `deployment/runtime/options.hpp`
- `deployment/runtime/point_features.cpp` (신규)
- `deployment/runtime/point_features.hpp` (신규)
- `deployment/runtime/run_inference.py`
- `deployment/runtime/tests/point_features_test.cpp` (신규)
- `deployment/runtime/tests/test_point_features.py` (신규)
- `deployment/runtime/tests/test_run_inference.py`
- `deployment/tensorrt/lidar_builder.py`
- `mmdet3d/models/backbones/dsvt_core.py`
- `new-arch.md`
- `tools/dsvt_pretrained/README.md`
- `tools/dsvt_pretrained/SPEC_phase1b.md` (작업 기준 문서, 시작 시 미추적)
- `tools/dsvt_pretrained/EXEC_phase1b.md` (신규)
- `tools/dsvt_pretrained/tests/test_official_layout.py`
