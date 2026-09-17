# 공식 DSVT 체크포인트 이식 실행 기록

## 1. 구현 전 확인

- 작업 브랜치: `dsvt-pretrained-init`
- 시작 시 기존 추적 파일 변경: 없음
- 시작 시 미추적 입력물: `pretrained/DSVT_Nuscenes_val.pth`, `tools/dsvt_pretrained/SPEC.md`, `tools/dsvt_pretrained/codex_run.log`
- main 기준 `mmdet3d/models/backbones/dsvt_core.py` SHA-256와 작업 트리 파일 SHA-256가 모두 `ff7fa6659c1b1283ba9332ca7d7e8af9b1e83b058da7a5086d7498e92bf99c88`로 동일했다.
- 공식 체크포인트 SHA-256: `a675149d095eef8ddc0c137ae46eeac075ccc504c7608162c71e7adf318793fb`
- 공식 체크포인트 확인: `model_state` 449 tensors, 7,130,122 tensor elements. 이 중 VFE 17,858, backbone_3d 1,202,184, backbone_2d 5,011,990 tensor elements다.
- 공식 config 확인: `LAYER_NUMS=[1,2,2]`, `LAYER_STRIDES=[1,2,2]`, `NUM_FILTERS=[128,128,256]`, `UPSAMPLE_STRIDES=[0.5,1,2]`, `NUM_UPSAMPLE_FILTERS=[128,128,128]`.
- `DSVTBEVFusion`이 LiDAR encoder를 `self.encoders["lidar"]["backbone"]`에 넣으므로 SPEC의 출력 checkpoint prefix `encoders.lidar.backbone.`가 실제 모델 경로와 일치한다.
- SPEC 가정 중 코드로 반증된 항목: 없음. 따라서 SPEC의 옵션 추가 원칙대로 `official_layout=False` 기본 경로를 보존하고 최소 변경으로 진행한다.

### 테스트 중 추가로 확인된 SPEC 가정 차이

- 최초 T3 실행에서 공식 `SetAttention`과 기존 구현의 출력 max-abs-diff가 `0.467746198`이었다.
- 원인: set padding으로 같은 pillar index가 반복될 때 공식 `dsvt.py`의 reverse 후 `scatter_` 코드는 결과적으로 **첫 occurrence**를 고르는 반면, 기존 `last_occurrence_gather()`는 마지막 occurrence를 고른다. 따라서 SPEC §1의 set attention “동일” 가정은 중복 padding 위치에서 코드로 반증되었다.
- 최소 수정: `official_layout=True`에서만 공식 first-occurrence gather를 사용하고, `False`에서는 기존 `last_occurrence_gather()`를 그대로 사용한다. 이로써 기존 경로의 key, shape, 수치를 보존한다.
- PyTorch 1.12의 `load_state_dict(strict=False)`는 누락된 BatchNorm `num_batches_tracked`를 `missing_keys`에 넣지 않는다. report에는 adapter state 6개를 모두 missing으로 기록하되, 실제 load 검증은 PyTorch가 반환하는 5개 missing key가 모두 `neck.adapter.*`인지 확인한다.

## 2. 실행 환경

- Python에서 확인한 PyTorch: `1.12.0+cu116`
- 테스트 방침: CPU만 사용

## 3. 실행 명령과 결과

### 체크포인트 조사

```bash
python - <<'PY'
# torch.load(..., map_location='cpu') 후 model_state key/shape/numel 집계
PY
```

- top-level key: `model_state`
- 전체: 449 tensors, 7,130,122 tensor elements
- VFE: 12 tensors, 17,858 elements
- backbone_3d: 192 tensors, 1,202,184 elements
- backbone_2d: 132 tensors, 5,011,990 elements
- dense_head: 112 tensors, 898,089 elements
- 기타: `global_step` 1 tensor/element

### 변환

```bash
python tools/dsvt_pretrained/convert_official_dsvt.py
```

출력 요약:

```text
converted matched=336 missing=6 unexpected=113 elements=6232032
checkpoint=.../pretrained/dsvt_nuscenes_official_lidar.pth
report=.../pretrained/dsvt_nuscenes_official_lidar.report.json
```

`report.json` 핵심 수치:

- matched: 336 keys, 공식 state tensor 6,232,032 elements
- 실제 학습 parameter에 해당하는 이식 elements: 6,222,144
- initialized: 3 keys, 12 elements (`vfe.voxel_size`, `point_cloud_range`, `grid_size`; 결정적 geometry buffer)
- missing: 6 keys, 전부 `encoders.lidar.backbone.neck.adapter.*`
- unexpected: 113 keys, 전부 `dense_head.*` 112개와 `global_step` 1개
- 공식 checkpoint element 이식 비율: 0.8740428284 (87.4043%)
- 우리 official-layout encoder state 초기화 비율: 0.9843116177 (98.4312%)
- 출력 checkpoint: 339 state tensors, 6,232,044 elements (공식 이식분 + geometry buffer)

### Config 검증

PyYAML로 원본/신규 config를 재귀 비교했다.

```text
config semantic diff
  model.encoders.lidar.official_layout: <missing> -> true
  model.encoders.lidar.zero_feature_channels: <missing> -> [4]
```

원본 `dsvt_dgf_dal_widthformer_0p3.yaml`에는 변경이 없다. 신규 config 끝에는
기본 schedule을 바꾸지 않는 주석 형태의 LiDAR `lr_mult: 0.1` 선택 예시만 있다.

### T1~T4 CPU 테스트

최종 실행 명령:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
pytest -q -s tools/dsvt_pretrained/tests/test_official_layout.py
```

최종 결과: **7 passed**, 3 warnings, 6.90초. warning은 PyTorch 1.12
`scatter_reduce_` beta 경고와 공식 원문의 `np.int` deprecation 경고뿐이다.

- **T1 PASS**: legacy `official_layout=False` state dict 263 keys와 모든 shape가
  main snapshot과 일치했다. main 원문을 `git show`로 임시 module에 import해 같은
  가중치/입력으로 전체 encoder forward를 비교했고 max-abs-diff `0`, bitwise
  equality를 확인했다.
- **D3 PASS**: `zero_feature_channels=[4]` 입력과 사용자가 직접 5번째 channel을
  0으로 만든 입력의 VFE feature/coordinate 출력이 bitwise 동일했다.
- **T2 PASS**: 변환 checkpoint를 official-layout encoder에 `strict=False`로
  로드했다. unexpected `0`; PyTorch가 반환한 missing `5`개는 모두 adapter다.
  BatchNorm `num_batches_tracked`는 PyTorch 1.12가 missing으로 반환하지 않으므로
  report의 전체 미초기화 adapter state는 6개다.
- **T3 PASS**: 공식 원문을 외부 읽기 전용 경로에서 최소 stub import했다.
  SetAttention max-abs-diff `0`, DSVT_EncoderLayer `0`, BasicBlock `0`,
  BaseBEVResBackbone `0`이었다.
- **T3 VFE 생략**: `torch_scatter`가 설치되어 있지 않고 공식 VFE 생성자가 CUDA
  tensor를 강제 생성한다. 패키지를 설치하지 말라는 지침에 따라 공식 VFE 직접
  동치 테스트는 생략했다. 대신 D3 zero-channel 회귀는 순수 torch fallback으로
  통과했다.
- **T4 대체 smoke PASS**: 변환 checkpoint를 로드한 official-layout
  `DSVTLidarEncoder`를 합성 5-channel points로 CPU forward했다. 출력 shape
  `(1, 256, 180, 180)`, 모든 값 finite를 확인했다.

표준 full-model smoke 시도:

```bash
python tools/smoke_train.py \
  configs/nuscenes/det/transfusion/secfpn/lidar/dsvt_dgf_dal_widthformer_0p3_dsvtpre.yaml \
  --forward-only
```

결과: `ModuleNotFoundError: No module named 'mmcv'`. 현재 환경에는 `mmcv`,
`torchpack`, `mmdet`, `nuscenes`, `spconv`, `torch_scatter`가 없다. 지침대로
설치하지 않았으며, 따라서 전체 `DynamicBEVFusion` build/forward/backward는
생략했다. 기존 `tools/smoke_train.py` 자체도 CUDA를 요구하므로 CPU full-model
경로가 아니다.

### T5 문서·git

- `tools/dsvt_pretrained/README.md`: 공식 model zoo 다운로드 위치, hash 확인,
  변환, CPU 테스트, `--load_from` 학습 명령을 기록했다.
- `NOTICE`: DSVT 출처와 Apache-2.0 고지를 추가했다.
- `new-arch.md`: “공식 DSVT 초기화” 절을 추가했다.
- `.gitignore`: `pretrained/*.pth`, `pretrained/*.json`을 추가했다. 입력 및 변환
  checkpoint/report는 로컬에는 존재하지만 commit 대상에서 제외된다.
- `git diff --check`: PASS.
- 커밋 identity: `WK-Hanelso <wk.hanelso@gmail.com>`
- 커밋 메시지: `dsvt-pretrained: 공식 DSVT nuScenes 체크포인트 LiDAR 분기 이식 (official_layout, converter, tests)`
- push: 수행하지 않음.

커밋을 위해 명시된 작업 파일만 `git add`하려 했으나, 실행 환경이 `.git`을
읽기 전용으로 마운트해 다음 오류로 실패했다.

```text
fatal: Unable to create '/home/hanelso/hanelso/bevfusion/.git/index.lock': Read-only file system
```

따라서 staging/commit은 생성되지 않았다. 작업 파일은 모두 working tree에
남아 있으며, checkpoint와 report는 `.gitignore` 적용 상태다.

## 4. 미완료 항목

- 의존 패키지와 nuScenes data가 있는 학습 환경에서의 전체 모델 1-step
  forward/backward는 위 사유로 미실행이다.
- 공식 VFE 직접 동치는 `torch_scatter`/CUDA 의존성 때문에 미실행이다.
- `.git` read-only 제약으로 지정 identity/message의 로컬 commit을 생성하지
  못했다. 쓰기 가능한 환경에서 동일 파일을 stage한 뒤 commit해야 한다.
- SPEC 범위 밖인 실제 nuScenes 학습과 Phase 2 head 이식은 수행하지 않았다.
