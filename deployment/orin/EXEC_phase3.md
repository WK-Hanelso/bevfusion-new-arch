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
