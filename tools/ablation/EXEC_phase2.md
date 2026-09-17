# Phase 2 실행 기록 — ablation config family + ModularBEVFusion

## 1. 시작 상태와 환경

- 시작 브랜치: `ablation-config-family`
- 시작 `git status --short --branch`: `## ablation-config-family`,
  `tools/ablation/`이 미추적 상태였다. 그 안의 작업 기준
  `SPEC_phase2.md`와 실행 로그 `codex_p2.log` 외에 추적 파일 변경은 없었다.
- `SPEC_phase2.md` 56줄 전체를 먼저 읽고 D1→D6, T1→T5 순서로 진행했다.
- Python 환경의 PyTorch는 `1.12.0+cu116`,
  `torch.cuda.is_available() == False`다.
- `mmcv`와 `torchpack`은 설치되어 있지 않았고 지침에 따라 설치하지 않았다.

## 2. 구현

### D1 — 모델 통일

- `ModularBEVFusion(BEVFusion)`을 추가했다. camera/LiDAR feature를 이름이
  있는 dict로 유지하고, inference의 encoder 실행 순서와 무관하게 fuser에는
  camera→LiDAR 순서로 전달한다.
- 순수 Python `select_fuser_call`, `select_head_call`을 별도 모듈로 분리했다.
  fuser는 `input_style`, head는 `needs_lidar_bev` 클래스 속성으로 dispatch한다.
  구체 클래스 `isinstance` 분기는 없다.
- `ConvFuser`/`AddFuser`는 `input_style = "list"`, `DepthGFusion`은
  `input_style = "named"`, `DALDecoupledHead`는
  `needs_lidar_bev = True`를 선언한다. 속성이 없는 fuser의 기본은 `list`다.
- 단일 센서는 fuser를 우회한다. DAL은 lidar-only에서 LiDAR encoder 출력을,
  camera-only에서 decoder 출력을 회귀 입력으로 받는다.
- legacy voxelized LiDAR config와 direct raw-point DSVT config를 모두 구성한다.
  학습 loss, 평가 bbox/map, depth auxiliary loss 처리는 기존 BEVFusion 경로와
  동일하게 유지했다.
- `DynamicBEVFusion`은 `ModularBEVFusion`의 얇은 registry alias로 바꿨다.
  기존 full config의 module/state key 구조와 camera-BEV seam은 유지된다.
  `dsvt_bevfusion.py`는 수정하지 않았다.

### D2/D3 — 공통 B0와 leaf 8개

- `configs/nuscenes/det/ablation/default.yaml`에 ResNet-34 + stride-16 LSSFPN,
  DepthLSS, voxel 0.075 SparseEncoder, ConvFuser, TransFusionHead B0와 공통
  single-sweep/no-GT-Aug pipeline, 6-camera, CBGS, AdamW/cyclic, 20 epochs를
  정의했다.
- Torchpack이 dict를 재귀 병합할 때 direct DSVT leaf에 SparseEncoder 전용
  key가 남지 않도록 legacy LiDAR subtree를 `lidar_legacy`에 두고
  `model.encoders.lidar: ${lidar_legacy}` scalar 참조로 연결했다.
  `recursive_eval` 후 B0에는 원래 voxelize/backbone dict가 전달된다.
- 8개 leaf는 변경 축만 담는다. DSVT leaf는 voxel 0.3과 head grid/factor를
  함께 바꾸며 `load_from`은 넣지 않았다.
- 순수 YAML 재귀 병합 결과 `c4_full`의
  `model.encoders/fuser/heads/decoder`는 기존
  `dsvt_dgf_dal_widthformer_0p3.yaml`의 같은 네 블록과 동일하다.
  기존 config 파일은 수정하거나 삭제하지 않았다.

### D4 — 생성기

- `gen_configs.py`가 lidar/vtransform/fuser/head 네 축의 Python dict와 8개
  조합 표에서 leaf를 생성한다.
- 파일마다 생성기 경로와 조합을 설명하는 결정적 comment header를 넣고
  생성 시각은 넣지 않았다. `--check`는 byte drift와 unified diff를 보고한다.

### D5 — 서버 smoke

- `smoke_all.sh`가 8개 config를 `tools/smoke_train.py`로 순차 실행한다.
  실패가 있어도 나머지를 계속 실행하고, 각 결과와 마지막 출력 줄을
  `smoke_results.md` 표에 append한 뒤 하나라도 실패하면 nonzero로 끝난다.
- `DATAROOT`와 `DEVICE` 환경 변수를 지원한다. DSVT smoke는 checkpoint 없이
  구조/gradient를 검사하며, 공식 초기화 학습은 README의 명시적
  `--load_from` 명령으로 분리했다.

### D6 — 문서와 원장

- `tools/ablation/README.md`에 8개 조합 표, B0와 원 BEVFusion의 차이,
  일반 학습 및 DSVT 4조합 각각의 공식 초기화 명령, 생성/테스트/smoke 명령,
  단일 센서 dispatch 계약을 기록했다.
- `new-arch.md`에 ModularBEVFusion 및 Phase 2 절을 추가했다.
- `experiments/LEDGER.md`에 지정된 정확도/초기화/data/sweeps/epoch와
  Thor·Orin Engine A/B/C·E2E P50 열 및 8개 빈 실험 행을 만들었다.

## 3. DoD 실행 결과

### T1 — 생성 결과 drift

```bash
python tools/ablation/gen_configs.py --check
```

```text
checked 8 ablation leaf configs
```

PASS: 커밋 대상 leaf 8개와 생성 결과가 byte-identical이다.

### T2/T3 — config 병합과 dispatch

```bash
pytest -q tools/ablation/tests
```

```text
................................                                         [100%]
32 passed in 2.77s
```

- T2는 Torchpack 0.3.1과 같은 디렉토리 `default.yaml` 순서 및 재귀 dict
  병합을 순수 PyYAML로 구현한다. c4/reference 네 모델 블록의 `${...}`는
  평가하지 않은 문자열 상태로 비교했다.
- 8개 조합의 module type, LiDAR별 head factor, 6-camera/single-sweep/
  no-GT-Aug/CBGS/20-epoch 공통 계약과 leaf의 `load_from` 부재도 확인했다.
- T3는 list/named fuser × standard/DAL head 네 조합을 fusion, lidar-only,
  camera-only 각각에 적용한 12개 경로를 fake module로 확인했다. 기본 list
  dispatch와 multi-sensor/no-fuser 오류도 확인했다.

### T4 — smoke script 문법과 정적 검사

```bash
bash -n tools/ablation/smoke_all.sh
python -m py_compile \
  tools/ablation/gen_configs.py \
  tools/ablation/tests/test_configs.py \
  tools/ablation/tests/test_dispatch.py \
  mmdet3d/models/fusion_models/dispatch.py \
  mmdet3d/models/fusion_models/modular_bevfusion.py \
  mmdet3d/models/fusion_models/dynamic_bevfusion.py
git diff --check
```

모두 PASS했다.

### T5 — 문서

조합 표, B0 차이, DSVT 초기화, 실행 명령, 단일 센서 계약, 아키텍처 절과
실험 원장을 추가했다. 실제 수치나 실행하지 않은 smoke 결과는 기록하지 않았다.

## 4. 로컬에서 실행하지 못한 검증

`mmcv`와 `torchpack`이 없고 CUDA를 사용할 수 없어 model package import,
registry build, nuScenes batch의 실제 8개 forward/backward/optimizer step은
실행하지 않았다. 이를 검증됐다고 간주하지 않는다. CUDA server에서 다음을
실행해야 한다.

```bash
DATAROOT=/data/nuscenes DEVICE=0 bash tools/ablation/smoke_all.sh
```

결과는 `tools/ablation/smoke_results.md`에 append된다. full 학습, exporter의
조합 지원, Thor/Orin 측정은 Phase 2 범위가 아니므로 실행하지 않았다.

## 5. git staging

작업 파일을 명시해 `git add`를 실행했으나 `.git`이 read-only라 staging은
생성되지 않았다. 지침대로 실패를 무시했으며 commit은 실행하지 않았다.

```text
fatal: Unable to create '/home/hanelso/hanelso/bevfusion/.git/index.lock': Read-only file system
```

`codex_p2.log`는 시작 시 존재하던 실행 로그이므로 staging 대상에서 제외했다.
모든 구현 변경은 working tree에 남아 있다.

## 6. Phase 2b — 전체 16개 조합과 테스트 격리

### 구현

- `gen_configs.py`가 DSVT/WidthFormer/GFusion/DAL 네 bit의 Cartesian product로
  canonical leaf 16개를 생성하도록 확장했다. legacy 8개 이름은 canonical
  mapping에서 생성되는 alias로 유지하고, 각 alias header에 canonical 파일명을
  기록했다.
- LiDAR 선택에서 voxel size, head train/test grid와 out-size factor, bbox coder
  factor가 함께 파생되게 했다. Sparse는 0.075/[1440,1440,41]/8, DSVT는
  0.3/[360,360,1]/2다.
- config 테스트는 canonical 16개와 alias 8개 전체의 파일 집합, 모듈 선택,
  geometry, 공통 실험 계약, alias YAML 동등성과 canonical header를 검사한다.
- `smoke_all.sh` 대상을 alias가 아닌 canonical 16개로 바꿨다. README에 전체
  조합 표와 alias 표를 추가하고, 원장을 canonical 16행으로 확장했다.
- 원장 지시의 "1차 9개"와 괄호 안 목록은 개수가 맞지 않았다. 별도 9번째
  조합이 지정되지 않았으므로 명시된 B0, E1~E4, C2~C4 대응 8개에 `1차`를
  표시했다.
- legacy forward 기준은 Phase 1b 이후의 `main` 대신 파일 상단 상수
  `LEGACY_BASE_REVISION = "829c7ae"`로 고정했다. candidate는
  `official_layout=False`를 명시하며 모델 동작 코드는 변경하지 않았다.

### 생성기·정적 검사

```bash
python3 tools/ablation/gen_configs.py --check
bash -n tools/ablation/smoke_all.sh
git diff --check
```

모두 PASS했다. 생성기 출력은 다음과 같다.

```text
checked 16 canonical and 8 alias ablation leaf configs
```

### 테스트

```bash
PYTHONPATH=. python3 -m pytest -q tools/ablation/tests
```

```text
74 passed in 8.47s
```

legacy 회귀 테스트 단독 실행도 통과했다.

```bash
PYTHONPATH=. python3 -m pytest -q \
  tools/dsvt_pretrained/tests/test_official_layout.py::test_legacy_forward_bitwise_equal
```

```text
1 passed, 1 warning in 3.20s
```

요청한 정순과 역순 통합 실행은 모두 통과했다.

```bash
PYTHONPATH=. python3 -m pytest -q \
  tools/ablation/tests tools/dsvt_pretrained/tests deployment/runtime/tests
```

```text
105 passed, 3 warnings in 17.98s
```

```bash
PYTHONPATH=. python3 -m pytest -q \
  deployment/runtime/tests tools/dsvt_pretrained/tests tools/ablation/tests
```

```text
105 passed, 3 warnings in 18.66s
```

경고는 기존 `scatter_reduce()` beta 경고 1건과 외부 official DSVT의
`np.int` deprecation 관련 2건이다.

### git staging

Phase 2/2b 산출물을 명시해 `git add`를 다시 시도했으나 `.git`이 read-only라
이전과 동일하게 staging되지 않았다. commit은 실행하지 않았다.

```text
fatal: Unable to create '/home/hanelso/hanelso/bevfusion/.git/index.lock': Read-only file system
```
