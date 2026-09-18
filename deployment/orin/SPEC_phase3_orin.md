# SPEC Phase 3 — DSVT LiDAR Engine B를 Orin(TensorRT 8.5)에서 실행: plugin·builder 8.x 호환 포팅

브랜치 `orin-trt85-plugins` (ablation-config-family a7709c0에서 분기). 사용자 요구(2026-09-18): "DSVT 오린에서 돌려야 해."

## 1. 사실
- Orin: Jetson AGX Orin, JetPack 5.1.2 / L4T R35.4.1, 컨테이너 `dustynv/l4t-pytorch:r35.4.1` = **CUDA 11.4 nvcc, TensorRT 8.5.2.2**(헤더 `/usr/include/aarch64-linux-gnu/NvInfer.h`, lib `/usr/lib/aarch64-linux-gnu/libnvinfer.so.8.5.2`), cmake 3.27.9, g++ 9.4, CUB 1.12.1, compute capability **8.7**. 15W 전력 모드(변경 금지).
- Thor 대상 현재 코드: plugin 5개(`deployment/tensorrt/plugins/*.cu`, 1,110줄)가 `IPluginV3`/`IPluginV3One{Core,Build,Runtime}`/`IPluginCreatorV3One`(TRT 10 전용). 출력 shape는 전부 **고정 용량**(MAX_POINTS 컴파일 옵션, pillar 10,000, set 512, coord 360×360) → 데이터 종속 shape 없음 → `IPluginV2DynamicExt`로 표현 가능.
- python builder `deployment/tensorrt/lidar_builder.py`: `plugin_registry.get_creator(...)`(TRT 10) → TRT 8.x는 `get_plugin_creator(name, version, namespace)`. `create_plugin(name, fc)`은 8.x에서 `create_plugin(name, fc)` 동일. `set_memory_pool_limit`, `set_input_shape`, `get_tensor_shape`는 8.5에 존재.
- C++ runtime `deployment/runtime/`: `enqueueV3`, `setTensorAddress`, `getNbIOTensors`, `getTensorShape`, `getTensorDataType`, `setInputShape`는 **TRT 8.5에 존재** → 원칙적으로 컴파일 가능. 확인 항목: `createInferRuntime` 시그니처, plugin 라이브러리 로드 방식(`dlopen` vs `IPluginRegistry::loadLibrary`(10 전용)).
- CMake `deployment/tensorrt/plugins/CMakeLists.txt`: `THOR_CUDA_ARCHITECTURES=110` 고정 캐시 변수.

## 2. 설계 결정
- **D1 단일 소스, 버전 분기**: 각 plugin `.cu`에서 `#if NV_TENSORRT_MAJOR >= 10` → 기존 V3 구현 그대로, `#else` → `IPluginV2DynamicExt` + `IPluginCreator` 구현. CUDA 커널·workspace 계산·용량 상수는 **공유**(중복 금지, 커널 함수는 버전 분기 밖에 둔다). plugin name/version 문자열, field 이름, 입출력 순서·dtype·shape는 두 경로가 완전히 동일해야 한다(builder가 버전을 몰라도 되게).
- **D2 V2DynamicExt 매핑**: `getOutputDimensions(index, inputs, nbInputs, exprBuilder)`, `supportsFormatCombination(pos, inOut, nbInputs, nbOutputs)`, `configurePlugin(in, nbIn, out, nbOut)`, `getWorkspaceSize(in, nbIn, out, nbOut)`, `enqueue(inDesc, outDesc, inputs, outputs, workspace, stream)`, `getOutputDataType`, `getSerializationSize/serialize`(필드 값 직렬화), `clone`, `initialize/terminate/destroy`, `setPluginNamespace/getPluginNamespace`, `getPluginType/getPluginVersion`. creator: `IPluginCreator` with `createPlugin(name, fc)`, `deserializePlugin(name, data, len)`, `getFieldNames`.
- **D3 CMake**: `THOR_CUDA_ARCHITECTURES` 기본값 유지하되 캐시 변수로 `87` 지정 가능해야 하고, TRT 버전을 헤더에서 읽어 로그로 출력. CUDA 11.4 + C++17 컴파일 가능해야 함(C++20 기능 사용 금지). CUB 1.12 API 범위 내.
- **D4 python builder**: `_get_plugin_creator(registry, name, version)` 헬퍼로 `get_creator`(10) / `get_plugin_creator`(8) 분기. TRT 8.5에서 LayerNorm ONNX op(opset 17)은 미지원 → `export_opset`이 16 이하이거나 LayerNorm이 분해되는지 `deployment/onnx/export_lidar_trt_artifacts.py`에서 확인하고, 아니면 8.x 경로용 opset 옵션을 추가(기본 동작 불변). ONNX parser·builder flag 차이(`BuilderFlag.TF32`는 8.5에 존재)는 try/except가 아니라 버전 확인으로 분기.
- **D5 C++ runtime**: TRT 8.5로 빌드해 보고 실패 지점만 최소 수정(`#if` 분기). plugin 로드는 `dlopen`(양쪽 공통)로 통일 가능하면 통일.
- **D6 Orin 빌드·검증 스크립트** `deployment/orin/build_plugins_orin.sh`: 컨테이너 안에서 `cmake -DTHOR_CUDA_ARCHITECTURES=87 -DDYNAMIC_BEVFUSION_MAX_POINTS=100000` 빌드 → `.so` 5개. `deployment/orin/plugin_unit_check.py`: TRT python으로 plugin 5개를 각각 단일 노드 네트워크로 등록·빌드·실행(랜덤/합성 입력, 고정 용량 shape)하고 (a) 빌드 성공 (b) 출력 finite (c) 가능하면 PyTorch 참조 구현(`dsvt_core.py`의 VFE/scatter/set partition)과 결과 비교. 참조 비교가 어려운 plugin(tensor_barrier)은 실행 성공만.
- **D7 문서**: `deployment/orin/README.md`(Orin 환경, 빌드·검증 명령, Thor와의 차이표, 미검증 항목), `deployment/tensorrt/README.md`에 8.x 지원 절 링크.

## 3. 작업 방식 (오케스트레이터 ↔ codex 라운드)
codex는 노트북에서 코드만 쓴다(컴파일 불가: x86, TRT 없음). 오케스트레이터가 Orin으로 동기화해 컨테이너에서 빌드·실행하고 **오류 로그 전문을 `deployment/orin/build_logs/roundN.log`로 저장**해 다음 라운드 입력으로 준다. 라운드 상한 4회. 각 라운드 codex는 `deployment/orin/EXEC_phase3.md`에 변경·근거를 append.

## 4. DoD
- Orin 컨테이너에서 plugin 5개 `.so` 빌드 성공(로그 첨부).
- `plugin_unit_check.py` 5개 통과(실행 성공 + finite; 참조 비교 가능한 것은 max-abs-diff 기록).
- Thor 경로(TRT ≥10) 소스 동작 불변: V3 코드 블록 diff 최소, 기존 CPU 계약 테스트(`deployment/runtime/tests`, launcher 18개 등) 통과.
- 후속(별도): LiDAR ONNX export(2060) → Orin Engine B 빌드·latency → A/B/C 전체.

## 5. 금지
- V3 경로 동작 변경 금지. 커널 수정 금지(포팅은 API 껍데기만). 용량 상수 변경 금지. Orin 전력 모드 변경 금지.
