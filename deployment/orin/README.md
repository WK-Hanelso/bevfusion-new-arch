# Jetson AGX Orin / TensorRT 8.5 배포

[TensorRT 공통 빌드](../tensorrt/README.md) · [Phase 3 SPEC](SPEC_phase3_orin.md)

대상은 JetPack 5.1.2 / L4T R35.4.1, CUDA 11.4, TensorRT 8.5.2.2,
compute capability 8.7인 Jetson AGX Orin이다. 검증 컨테이너는
`dustynv/l4t-pytorch:r35.4.1`이며 15W 전력 모드는 변경하지 않는다.

## Thor와의 차이

| 항목 | Orin | Thor |
|---|---|---|
| TensorRT plugin API | `IPluginV2DynamicExt` / `IPluginCreator` | `IPluginV3One*` / `IPluginCreatorV3One` |
| TensorRT | 8.5.2.2 | 10.13.2.6 |
| CUDA | 11.4 | 13.0 |
| CUDA architecture | 87 | 110 |
| plugin 등록 | `get_plugin_creator`, `add_plugin_v2` | `get_creator`, `add_plugin_v3` |
| plugin 로드 | `.so`를 `dlopen`/`ctypes.CDLL` | 동일 |

Plugin 이름/version, field, 입출력 순서·dtype·shape와 capacity는 양쪽에서
동일하다. `.cu` 파일은 TensorRT major version으로 V2/V3 껍데기만 분기하고
CUDA 커널과 workspace/enqueue 구현을 공유한다.

## 컨테이너에서 plugin 빌드

Repository를 컨테이너의 `/workspace/bevfusion`에 mount했다고 가정한다.

```bash
cd /workspace/bevfusion
bash deployment/orin/build_plugins_orin.sh
```

기본 설치 위치는
`deployment/orin/build/plugins-trt85-install`이다. 별도 위치가 필요하면
`BUILD_DIR`과 `INSTALL_DIR` 환경 변수를 지정한다. 스크립트는 다음 고정 계약으로
CMake를 실행한다.

```text
THOR_CUDA_ARCHITECTURES=87
DYNAMIC_BEVFUSION_MAX_POINTS=100000
DYNAMIC_BEVFUSION_MAX_PILLARS=10000
DYNAMIC_BEVFUSION_MAX_SETS=512
```

CMake configure 로그에서 읽은 TensorRT header version과 CUDA architecture를
확인한다. 산출물은 plugin `.so` 다섯 개다.

## 단일-node plugin 검증

```bash
python deployment/orin/plugin_unit_check.py \
  --plugin-dir deployment/orin/build/plugins-trt85-install
```

검사는 각 plugin을 별도 TensorRT network로 build/deserialize/enqueue하고 float
출력의 finite 여부를 확인한다. Decorate, scatter-max, rotated-set 위치/count,
dense-scatter 및 barrier는 합성 입력 참조값과 비교하고 `max_abs_diff`를 출력한다.

## Engine B artifact와 builder

LiDAR exporter의 기본 opset은 16이다. Opset 17의 native LayerNormalization을
사용하지 않으므로 TensorRT 8.5 경로에서도 기존 기본값 `--opset 16`을 유지한다.
Artifact 생성과 Engine B 명령은 [TensorRT 공통 문서](../tensorrt/README.md)의
절차를 따르되 plugin directory만 위 Orin 설치 위치로 지정한다. TensorRT 8.5는
builder optimization level 및 auxiliary-stream 수를 노출하지 않으므로 현재
builder는 기존 기본값 `--optimization-level 3 --max-aux-streams 0`만 허용한다.

## 아직 Orin에서 확인할 항목

- TensorRT 8.5.2.2 header로 plugin 다섯 개의 모든 `override`/`noexcept` 시그니처
- CUDA 11.4/CUB 1.12 및 `sm_87` 실제 컴파일과 link
- `plugin_unit_check.py` 다섯 network의 build, 실행, finite 및 참조 오차
- Engine B ONNX parse/build/deserialize와 C++ runtime build
- 실제 Orin latency, memory 및 정확도

Orin 실행 로그 전문은 `deployment/orin/build_logs/roundN.log`에 보관한다.
