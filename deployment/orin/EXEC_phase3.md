# EXEC Phase 3

## Round 1 — TensorRT 8.5 plugin/builder compatibility port

### 변경

- Plugin 5개에 `NV_TENSORRT_MAJOR >= 10` 분기를 추가했다. 기존 V3 creator와
  plugin 계약은 유지하고 TensorRT 8.x에서는 `IPluginV2DynamicExt` 및
  `IPluginCreator`를 등록한다.
- V2 경로에 output expression/dtype/format, configure, workspace, enqueue,
  zero-byte serialization/deserialization, lifecycle, clone 및 namespace API를
  구현했다. Plugin 이름/version, 입력·출력 순서, dtype, shape와 capacity는 V3와
  동일하다.
- CUDA kernel, capacity 상수, workspace 계산과 enqueue 본체는 버전 분기 밖의
  공용 구현을 사용한다. Kernel 계산 자체는 변경하지 않았다.
- Plugin CMake가 `NvInferVersion.h`의 TensorRT version을 읽어 configure 로그에
  출력한다. 기존 architecture 기본값 110은 유지하며 Orin script는 87을
  명시한다.
- Python builder에 TensorRT major-version 판별, `_get_plugin_creator`, V2/V3
  plugin 생성·layer 추가, TRT 8.5 explicit-batch network 분기를 추가했다.
  TensorRT 8.5에 없는 builder optimization-level/aux-stream 설정은 기존 기본값
  3/0만 허용하고 API 호출을 생략한다.
- LiDAR exporter는 이미 기본/강제 opset 16을 사용하므로 opset 17
  LayerNormalization 회피를 위한 변경은 하지 않았다.
- C++ runtime은 양쪽 모두 `.so`를 `dlopen`으로 로드한다. TensorRT 8.5에 없는
  `getDeviceMemorySizeV2`만 `getDeviceMemorySize`로 version 분기했다.
- Orin `sm_87` plugin build script, plugin 5개 단일-node build/execute/reference
  검사, 환경·명령·Thor 차이 문서를 추가하고 TensorRT README에서 연결했다.

### 로컬 검증

- `python -m py_compile deployment/tensorrt/lidar_builder.py deployment/tensorrt/builder.py deployment/orin/plugin_unit_check.py deployment/onnx/export_lidar_trt_artifacts.py`: PASS
- `PYTHONPATH=. pytest -q deployment/runtime/tests`: PASS (`20 passed`)
- `bash -n deployment/orin/build_plugins_orin.sh`: PASS
- TensorRT 8/10 mock registry 및 versioned builder-option helper 검사: PASS
- Plugin 5개 V2 필수 method/preprocessor/괄호 정적 계약 검사: PASS
- `git diff --check`: PASS

### 미확인 / Orin에서 확인 필요

- 로컬 x86 환경에 TensorRT/CUDA compiler와 header가 없어 `.cu` 및 C++ compile은
  수행하지 못했다. TensorRT 8.5.2.2 header가 V2DynamicExt/creator의 모든
  `override`와 `noexcept`를 수락하는지 첫 Orin build log에서 확인해야 한다.
- CUDA 11.4 + CUB 1.12.1 + `sm_87` compile/link 및 plugin `.so` 다섯 개 생성은
  미확인이다.
- `plugin_unit_check.py`의 TensorRT build/deserialize/enqueue, finite 검사 및
  reference `max_abs_diff`는 Orin에서 실행해야 한다.
- Engine B ONNX parser/build와 C++ runtime 전체 build는 후속 Orin 검증 대상이다.
- Orin 실행 로그 전문은 `deployment/orin/build_logs/round1.log`로 저장한다.

### Staging

- 관련 파일만 명시한 `git add`를 시도했으나 이 환경에서 `.git/index.lock` 생성이
  `Read-only file system`으로 거부되어 staging되지 않았다. Commit은 생성하지
  않았다.

## Round 2 — TensorRT 8.5 Python API compatibility and build isolation

### 원인과 변경

- Engine B frontend가 TensorRT 8.6에서 추가된 `INetworkDefinition.add_cast`를
  세 곳에서 직접 호출해 TensorRT 8.5.2.2에서 첫 호출 즉시 중단됐다.
- `trt.__version__`의 major/minor를 파싱하는 공용 helper를 추가하고 모든 cast를
  `_cast(network, tensor, dtype)`로 통일했다. TensorRT 8.6 이상은 `add_cast`를
  유지하고, 8.5는 `add_identity` 뒤 `set_output_type(0, dtype)`와 output tensor의
  `dtype` 지정을 함께 사용한다.
- `lidar_builder.py`, `builder.py`, `build_all.py`의 TensorRT Python 호출을 다시
  점검했다. `add_normalization`, `add_nms`, `add_grid_sample` 호출은 없다.
  현재 사용하는 named-I/O API와
  `IExecutionContext.set_input_shape/get_tensor_shape`는 8.5에 존재하므로 별도
  fallback이 필요하지 않다. 신규 `device_memory_size_v2`는 기존처럼
  `device_memory_size` fallback을 유지한다.
- `build_all.py`는 camera, raw-LiDAR, fusion 빌드를 각각 `try/except`로 격리한다.
  한 엔진의 예외와 traceback을 기록한 뒤 다음 엔진을 계속 빌드하며, bundle
  manifest의 `engines`에는 이번 실행에서 성공한 엔진만 기록한다. 모든 시도와
  manifest 기록 후 실패 엔진 이름/원인을 요약하고 exit code 1을 반환한다.

### TensorRT 8.5 API 근거

- NVIDIA TensorRT 8.5.3 Python API의 `INetworkDefinition`에는
  `add_identity`, `add_nms`, `add_grid_sample`이 있지만 `add_cast`와
  `add_normalization`은 없다. 두 API는 8.6.1 문서의 `INetworkDefinition`에
  나타난다.
- TensorRT 8.5.3 Python API의 `ILayer.set_output_type`과 `ITensor.dtype`은
  identity layer output의 계산/output dtype을 명시할 수 있다.
- TensorRT 8.5.3 Python API의 `IExecutionContext`에는
  `set_input_shape(name, shape)`와 `get_tensor_shape(name)`가 있고,
  `ICudaEngine`에는 `num_io_tensors` 및 `get_tensor_*` named-I/O 조회 API가 있다.
- 문서:
  [TRT 8.5.3 INetworkDefinition](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-853/api/python_api/infer/Graph/Network.html),
  [TRT 8.5.3 IExecutionContext](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-853/api/python_api/infer/Core/ExecutionContext.html),
  [TRT 8.6.1 INetworkDefinition](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/api/python_api/infer/Graph/Network.html).

### 로컬 검증

- `python -m py_compile deployment/tensorrt/*.py`: PASS
- `PYTHONPATH=. pytest -q deployment/runtime/tests`: PASS (`20 passed`)
- TensorRT 8.5.2.2/8.6.1/10.13.2.6 mock으로 `_cast`의 identity/cast 분기,
  dtype 지정 확인: PASS
- mock Engine B 실패로 A → B → C 순서의 세 빌드가 모두 시도되고, partial bundle에
  A/C만 포함되며 반환 코드가 1인지 확인: PASS
- `git diff --check`: PASS
- TensorRT가 로컬에 없으므로 실제 engine build는 Orin 재검증 대상으로 남긴다.

## Round 3 — Engine C ONNX conditional Squeeze 제거

### 원인과 변경

- `ContinuousPositionBias`의 마지막 `Linear(..., 1)` 출력에 적용한
  `squeeze(-1)`가 ONNX에서 `Shape → Gather → Equal → If`로 내려갔다. TensorRT
  8.5는 then=`Squeeze`, else=`Identity`인 `If_123`의 서로 다른 출력 rank를
  허용하지 않아 Engine C parsing이 실패했다.
- 마지막 축은 MLP 정의상 항상 1이므로 `self.mlp(relative)[..., 0]` 정적
  인덱싱으로 교체했다. 파라미터와 buffer를 변경하지 않아 기존 checkpoint 키는
  그대로다.
- depth-guided attention의 depth encoding은 shape 비교 없이 항상
  `expand_as`를 적용한다. 이미 같은 shape이면 view 수준의 no-op이므로 수치와
  gradient는 동일하다. eager 입력 shape 검사는 유지하되 ONNX export 중에는
  Python shape 분기를 건너뛴다.
- WidthFormer의 singleton matrix-vector 축 두 곳도 `[..., 0]`으로 바꾸고,
  semantic-attention chunk 조건은 tensor shape 대신 고정 BEV token 수를
  사용한다. `forward_with_geometry`의 eager shape 검사는 export 중 제외했다.
- DAL/TransFusion의 class-channel 조건은 tensor shape 대신 설정값
  `num_classes`를 사용한다. DAL의 fused/LiDAR grid 검사는 eager에서는 유지하고
  export 중에는 제외했다. 후처리의 `task_keep_indices.shape[0]` 조건은 Engine C
  wrapper가 호출하지 않는 bbox decode/NMS 경로라 변경하지 않았다.

### 로컬 검증

- 신규 순수-PyTorch CPU 회귀 테스트는 변경 전 squeeze 공식을 oracle로 삼아
  출력, query/sample 입력 gradient, MLP 전체 파라미터 gradient가 정확히 같고
  state-dict key 6개가 유지됨을 확인한다.
- `PYTHONPATH=. pytest -q tools/ablation/tests tools/dsvt_pretrained/tests deployment/runtime/tests`:
  PASS (`107 passed`; 기존 105 + 신규 2).
- 관련 5개 Python 파일 `python -m py_compile`: PASS.
- CPU에서 position-bias 모듈을 opset 16 ONNX로 export해 연산 종류를 검사한
  결과 `If=0`, `Squeeze=0`, 정적 축 선택 `Gather`가 존재했다.
- `git diff --check`: PASS.

### Orin 재검증 필요

- 갱신된 `fusion_dal.onnx`에서 `If` node가 0개인지 확인한 뒤 TensorRT 8.5.2로
  Engine C를 재빌드한다. 실제 Orin parser/build 결과는 로컬 환경에서 확인할 수
  없다.

## Round 4 — TensorRT 8.5 bool cast 제거

### 원인과 변경

- Round 2의 TensorRT 8.5 fallback은 모든 cast를 `add_identity`와
  `set_output_type`으로 구현했지만, TensorRT 8.5.2의 identity-cast는 bool
  입력/출력을 지원하지 않아 Engine B 빌드가 실패했다.
- `_cast`의 TensorRT 8.5 경로를 source/target dtype별로 분리했다. bool에서
  수치형으로 바꿀 때는 입력 rank와 같은 rank의 1/0 상수를 대상 dtype으로 만들고
  `add_select`를 사용한다. 수치형에서 bool로 바꿀 때는 0보다 큰 값과 작은 값을
  각각 비교한 뒤 bool `OR`로 합쳐 nonzero 판정을 만든다. 수치형끼리의 변환은
  기존 identity-cast를 유지하며, 같은 dtype은 원본 tensor를 그대로 반환한다.
- 변환용 상수는 `weight_store`에 보관해 engine build가 끝날 때까지 NumPy backing
  storage의 수명을 유지한다. TensorRT 8.6 이상 `add_cast` 경로는 변경하지 않았다.

### bool 사용처 점검

- `add_frontend`의 `valid`는 `GREATER` 결과 bool이며 float32 마스크로 변환된다.
  이제 `Select(valid, 1.0, 0.0)`으로 구성되어 bool identity-cast가 없다.
- `add_capacity_status`의 `pillar_overflow`와 `set_overflow`도 비교 결과 bool이며
  int32 status로 변환된다. 두 경로 모두 int32 1/0을 선택한다.
- `set_masks_shift_0/1`은 현재 DSVT backbone ONNX 입력과 plugin 출력이 모두
  int32라 cast가 없다. 그 밖에 `lidar_builder.py`에는 명시적인 bool 관련
  `add_identity`, `set_output_type`, `add_cast` 호출이 없다.

### 로컬 검증

- `python -m py_compile deployment/tensorrt/lidar_builder.py`: PASS
- `PYTHONPATH=. pytest -q deployment/runtime/tests`: PASS (`22 passed`)
- TensorRT mock으로 8.5의 bool→float32 `Select`, int32→bool nonzero 비교,
  수치형→수치형 identity, bool→bool no-op과 10.x `add_cast` 경로를 확인했다: PASS
- `git diff --check`: PASS
- 실제 TensorRT 8.5.2 Engine B build는 Orin 재검증 대상으로 남긴다.

### 라운드 4b (오케스트레이터 직접 수정)
- Orin 컨테이너 numpy 1.24.4에서 TensorRT 8.5 바인딩의 `trt.nptype()`이 `np.bool`을 참조해 실패 → `lidar_builder.py`·`builder.py` 상단에 `hasattr(np,"bool")` 보호 shim 추가. 우리 코드의 np.bool 사용은 0건.

## Round 5 — ONNX exporter CPU-only 진단 경로

### 원인과 변경

- 다섯 ONNX exporter는 `--device cpu`를 받아도 CUDA availability를 무조건
  검사해, CUDA context를 만들 수 없는 노트북에서 이미 검증된 CPU export 경로를
  사용할 수 없었다.
- 공통 `require_export_device` 검사를 추가했다. `--device cpu`와
  `--allow-cpu-only`를 함께 지정한 경우에만 CUDA 없이 진행하고, 그 밖의 경우에는
  기존 `CUDA is required for this deployment export` 오류를 유지한다.
- `export_all.py`가 새 플래그를 camera, raw-LiDAR, fusion 하위 exporter에
  전달한다. 생성된 manifest의 model provenance에는 실제 `export_device`와
  `cpu_only`가 기록되어 CPU-only 산출물을 진단용으로 식별할 수 있다.

### 로컬 검증

- `python -m py_compile`로 ONNX exporter와 공통 helper 문법 검사: PASS
- `PYTHONPATH=. pytest -q deployment/runtime/tests`: PASS (`22 passed`)
- CUDA unavailable mock에서 CPU+opt-in 허용, CPU opt-in 누락과 CUDA device 요청의
  기존 오류 메시지 유지, manifest provenance 값 생성을 확인: PASS
- `git diff --check`: PASS

## 라운드 6 — LiDAR padded backbone ONNX의 Cast→BOOL 제거

### 원인과 변경

- `PaddedDSVTBackbone`의 `masks0.bool()`/`masks1.bool()`가 graph 입력 직후
  `Cast(to=BOOL)` 2개를 만들고, PyTorch `MultiheadAttention`의
  `key_padding_mask` canonicalization이 8개 attention layer마다 Reshape 뒤
  같은 Cast를 하나씩 추가했다. 따라서 builder의 수동 bool cast를 제거한
  라운드 4와 별개로 padded backbone ONNX 자체에 bool 출력 Cast 10개가 남았다.
- `SetAttention.forward_with_gather`는 eager/학습에서 기존
  `nn.MultiheadAttention`을 그대로 사용한다. ONNX export 중에만 동일 weights로
  Q/K/V projection과 scaled dot-product attention을 명시적으로 계산하고,
  int32 padding mask의 `mask != 0` 결과로 `torch.where(..., -inf, scores)`를
  적용한다. 이 경로는 ONNX `Equal → Not → Where`로 내려가며 Cast로 bool을
  만들지 않는다.
- `DSVTDeployWrapper`의 graph 입력 mask 계약을 int32로 고정했다. eager wrapper는
  호출 시에만 bool로 바꿔 기존 수치 경로를 유지하고, export 중에는 int32를
  `SetAttention`까지 전달한다. `PaddedDSVTBackbone`의 입력 `.bool()` 두 곳은
  제거했다.
- LiDAR artifact manifest의 padded-backbone 항목에 모든 내부 입력 dtype/shape와
  `cast_to_bool_nodes: 0`을 기록한다. builder는 ONNX 입력, manifest, frontend/plugin
  출력의 dtype이 각각 일치하는지 연결 전에 검사한다. mask/index/gather는 int32,
  src/position embedding은 float32다.
- `DSVTRotatedSet` plugin은 기존부터 mask를 포함한 해당 출력을 int32로 내므로
  plugin source는 수정하지 않았다.
- 두 ONNX exporter는 생성 직후 `Cast(to=BOOL)` 존재 여부를 검사하고 하나라도
  있으면 artifact 생성을 실패시킨다.

### 수치 및 그래프 회귀

- 기존 공식/legacy wrapper 동치와 공식 `SetAttention` 동치 테스트는 그대로
  유지했다. 추가 CPU 테스트는 export 전용 attention과 eager
  `MultiheadAttention` 결과를 비교한다.
- 작은 2-block `DSVTDeployWrapper`를 CPU에서 opset 16으로 export해 두 mask 입력이
  ONNX INT32이고 `Cast(to=BOOL)`이 0개이며 비교 op가 존재하는지 검사하는 테스트를
  추가했다. `onnx` 패키지가 없으면 이 테스트 파일은 skip한다.
- 실제 4-block `PaddedDSVTBackbone`을 진단 capacity 48 pillars/48 sets로 CPU
  export한 결과: 920 nodes, `Equal=8`, `Not=8`, `Where=8`, `Cast=12`이며 12개
  Cast는 모두 INT64, `Cast(to=BOOL)=0`이었다. 두 mask graph 입력은 INT32였다.

### 로컬 검증

- 관련 Python 파일 `python -m py_compile`: PASS
- `PYTHONPATH=. pytest -q tools/dsvt_pretrained/tests deployment/runtime/tests`:
  PASS (`35 passed`; warning 3개는 기존 scatter/NumPy deprecation)
- `git diff --check`: PASS
- 로컬에 TensorRT 8.5/Orin이 없어 실제 Engine B parse/build는 재검증 대상으로
  남긴다. 변경 중 commit은 생성하지 않았다.

### 라운드 6b (오케스트레이터 직접 수정)
- torch 1.10 ONNX exporter가 `scores.new_full((), -inf)`(0차원 shape) 심볼릭에서 segfault(faulthandler: symbolic_opset9.full) → `torch.full_like(scores, -inf)`로 교체(ConstantOfShape). 수치 동일.

### 라운드 6c (오케스트레이터 직접 수정)
- `full_like(-inf)`가 상수 접기로 레이어별 [512,8,90,90] 상수로 박혀 ONNX 1.07GB → 산술 마스킹(`scores + mask_float * -1e4`, mask int32→float Cast만 사용)으로 교체. 재export 후 크기·Cast→BOOL 0 확인 예정.
