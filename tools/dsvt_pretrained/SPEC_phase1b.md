# SPEC Phase 1b — 공식 DSVT 구조를 기본 구조로 채택 + 배포 export 정합성

브랜치 `dsvt-pretrained-init`, Phase 1(커밋 68d23d0) 위에 작업. 사용자 결정(2026-09-17): "official_layout을 기본 구조로 가져간다."

## 1. 배경·근거 (사실)
- Phase 1 결과: 공식 DSVT_Nuscenes_val.pth의 LiDAR 분기 6,232,032 params가 `official_layout=True` 인코더에 1:1 로드됨(missing = neck.adapter만). 테스트 7 passed.
- Orin 실측(15W, FP16): BEV backbone TRT 47→78 ms(+65%), 인코더 eager 465→538 ms. 사용자는 이 비용을 수용하고 공식 구조를 기본으로 채택.
- **결함**: `deployment/onnx/dsvt_backbone.py`(52~57행)는 `block.layers`만 순회하고 `block.layer_norms`를 적용하지 않는다. `official_layout=True` 모델을 export하면 PyTorch와 다른 그래프가 된다. 이 SPEC의 최우선 항목.

## 2. 설계 결정
- **D1 기본값 전환**: `DSVTLidarEncoder`, `DSVTBackbone`, `DSVTBlock`, `SetAttention`의 `official_layout` 기본값을 `True`로. `DynamicPillarVFE.zero_feature_channels` 기본값은 `[]` 유지(config에서 명시).
- **D2 config 정리**:
  - `configs/.../dsvt_dgf_dal_widthformer_0p3.yaml` = **기본 구조**: `official_layout: true`, `zero_feature_channels: [4]` 명시. 주석에 "공식 DSVT 체크포인트 초기화 가능, `tools/dsvt_pretrained/README.md` 참조".
  - `configs/.../dsvt_dgf_dal_widthformer_0p3_legacy.yaml` 신규: `official_layout: false`, `zero_feature_channels: []`. 주석에 "2026-09-10 full 학습 ep19 mAP 0.5281 / NDS 0.5132 및 Thor 26.4 ms 측정에 사용된 구조. 재현 전용, 공식 체크포인트 로드 불가".
  - `..._0p3_dsvtpre.yaml`은 삭제(기본 config가 그 역할). README/new-arch.md의 참조를 갱신.
- **D3 export 정합성**:
  - `deployment/onnx/dsvt_backbone.py`: 블록 순회 시 `official_layout`이면 각 `layers[axis]` 뒤에 `layer_norms[axis](output + input)`을 적용. 인코더의 `official_layout` 값을 읽어 분기(하드코딩 금지). `forward_with_gather` 경로 유지.
  - `deployment/onnx/export_lidar_trt_artifacts.py`·`model_contract.py`: lidar manifest에 `official_layout`과 `zero_feature_channels`를 기록. neck wrapper는 `DSVTBEVResNeck`을 그대로 감싸는지 확인(forward 시그니처 동일).
  - **5번째 채널 0 처리의 배포 반영**: TensorRT 경로는 raw points를 plugin(`dynamic_pillar_decorate`)이 직접 처리하므로 PyTorch의 `zero_feature_channels`가 자동 반영되지 않는다. 최소 변경 원칙으로 **C++ runtime(`deployment/runtime`)에서 manifest의 `zero_feature_channels`를 읽어 H2D 전에 해당 채널을 0으로 채우는** 방식을 채택한다. plugin 코드는 수정하지 않는다. Thor 없이 검증 불가한 부분은 CPU 단위 테스트(채널 0 처리 함수)와 계약 테스트로 한정하고 EXEC.md에 "Thor 실측 미검증"으로 기록.
  - `deployment/README.md`·`deployment/runtime/README.md`의 입력 계약 표에 "feature[4]는 runtime이 0으로 덮음(기본 config)" 문장 추가. 기존 실측 수치(26.4 ms 등)는 legacy 구조 기준임을 명시.
- **D4 Phase 1 테스트 갱신**: `tests/test_official_layout.py`의 legacy 스냅샷 테스트는 `official_layout=False`를 명시 전달하도록 수정(기본값이 바뀌므로). 기본 생성자 = official 구조임을 확인하는 테스트 1개 추가.

## 3. DoD
- T1 기존 7개 테스트 통과(기본값 변경 반영 후).
- T2 export 정합성: `official_layout=True` 인코더에 변환 ckpt 로드 → `deployment/onnx/dsvt_backbone.py` 래퍼 출력과 `DSVTBackbone.forward` 출력 max-abs-diff == 0 (CPU, 합성 입력). legacy도 동일 테스트 유지.
- T3 neck 래퍼: `DSVTBEVResNeck` export 래퍼 출력 == 직접 호출 출력.
- T4 기존 deployment CPU 계약 테스트(launcher 18개 등, `deployment` 아래 pytest) 전부 통과. mmcv 부재로 못 도는 테스트는 목록과 사유를 EXEC.md에.
- T5 문서: `new-arch.md`(기본 구조 = 공식 DSVT 구조, legacy 절), `tools/dsvt_pretrained/README.md`, `deployment/README.md`, `deployment/runtime/README.md`.
- git: staging까지만(커밋은 오케스트레이터). 파일 목록을 EXEC.md 끝에.

## 4. 금지
- `transfusion.py`, `dal_decoupled.py`, camera/fusion 코드, TensorRT plugin 소스 수정 금지.
- legacy 경로의 수치 동작 변경 금지(T1 회귀로 보장).
- Thor에서 검증 안 된 latency 수치를 문서에 쓰지 않는다.

## 5. EXEC
`tools/dsvt_pretrained/EXEC_phase1b.md`에 명령·출력·테스트 결과·미검증 항목 기록.
