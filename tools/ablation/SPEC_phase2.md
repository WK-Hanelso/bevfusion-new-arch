# SPEC Phase 2 — 모듈 조합을 config로 선택하는 ablation config 계열 + 모델 클래스 통일

브랜치 `ablation-config-family` (main 6c16f68에서 분기). 사용자 결정(2026-09-17): "기본 BEVFusion이 있고 거기에 DSVT / WidthFormer / DAL / GFusion을 각각 추가하는 것을 각각 config로. 학습 코드 수정 없이 config 선택만으로." 목적은 논문용 ablation 사다리(정확도 + 엣지 latency)이며 모든 실험 기록은 config 이름으로 구분한다.

## 0. 범위
- **포함**: (A) 단일 모델 클래스로 legacy/신규 모듈의 모든 조합 실행, (B) config 계열 8개 + 공통 default, (C) config 생성기와 드리프트 테스트, (D) 서버용 smoke 스크립트, (E) 실험 원장 템플릿.
- **비포함**: 학습 실행, 배포 exporter의 조합 지원(Phase 3), Orin/Thor 측정.

## 1. 인터페이스 사실 (코드 확인 완료)
| 축 | legacy 모듈 | 신규 모듈 | 출력 | 비고 |
|---|---|---|---|---|
| LiDAR encoder | `SparseEncoder` voxel 0.075, sparse_shape [1440,1440,41] | `DSVTLidarEncoder` pillar 0.3, official_layout | 둘 다 BEV 256ch **180×180** | **head의 voxel_size / out_size_factor / grid_size가 encoder에 종속**: Sparse=0.075/8/[1440,1440,41], DSVT=0.3/2/[360,360,1] |
| Camera vtransform | `DepthLSSTransform` (80ch) | `WidthFormerTransform` (80ch) | 80ch 180×180 | DepthLSS는 GTDepth 파이프라인·depth loss 옵션 사용 가능 |
| Fuser | `ConvFuser(inputs: List)` in [80,256] → 256 | `DepthGFusion(camera_bev, lidar_bev)` → 256 | 256ch 180×180 | **호출 시그니처가 다름** |
| Head | `TransFusionHead.forward(feats, metas)` | `DALDecoupledHead.forward(fused, lidar_bev, metas)` | — | **DAL은 lidar BEV를 별도로 받음** |
| 모델 클래스 | `BEVFusion.forward_single`: features list → fuser(list) → decoder → head(x, metas) | `DynamicBEVFusion`: fuser(cam, lidar) → decoder → head(x, lidar_bev, metas) | — | 통일 대상 |
| Camera backbone | legacy config는 ResNet-50 + GeneralizedLSSFPN(stride 8) | 신규는 ResNet-34 + LSSFPN(stride 16) | — | §2 D2 |
| Config 로더 | torchpack `configs.load(recursive=True)`: 디렉토리 경로상의 `default.yaml`을 순서대로 병합. `_base_` 없음 | | | 계열은 디렉토리 + leaf 파일로 표현 |

## 2. 설계 결정
- **D1 모델 클래스 통일** — `mmdet3d/models/fusion_models/modular_bevfusion.py`에 `ModularBEVFusion(BEVFusion)` 신설. 규칙:
  - 센서 특징을 `dict{"camera": bev, "lidar": bev}`로 모은다. 없는 센서는 생략(lidar-only / camera-only 허용).
  - fuser 호출: 모듈이 `forward(inputs: List)`(ConvFuser/AddFuser)이면 `[camera, lidar]` 순서의 list, `forward(camera_bev, lidar_bev)`(DepthGFusion)이면 named 호출. 판별은 isinstance 분기 대신 fuser 클래스 속성 `input_style = "list" | "named"`로(ConvFuser/AddFuser/DepthGFusion에 클래스 속성 1줄씩 추가, 기본 "list"). fuser가 None이면 단일 센서 특징을 그대로.
  - head 호출: head 클래스 속성 `needs_lidar_bev = True`(DALDecoupledHead)면 `head(x, lidar_bev, metas)`, 아니면 `head(x, metas)`. lidar-only에서 DAL은 `lidar_bev = lidar 특징`, camera-only에서 DAL은 `lidar_bev = decoder 출력`(회귀 입력 대체)로 정의하고 문서화.
  - 학습 손실·평가·auxiliary loss(depth) 경로는 `BEVFusion`의 것을 그대로 상속. `DynamicBEVFusion`은 `ModularBEVFusion`의 얇은 alias로 남겨 기존 config·체크포인트 호환(클래스명 유지, 동작 동일). `dsvt_bevfusion.py`는 건드리지 않는다.
- **D2 공통 baseline B0 고정** — camera backbone을 **ResNet-34 + LSSFPN(in_indices [2,1], stride 16)** 으로 전 조합 고정한다(엣지 타깃, 신규 모델과 동일). legacy vtransform `DepthLSSTransform`은 stride 16 feature_size(`image_size // 16`)로 설정. B0 = ResNet-34 + DepthLSS + SparseEncoder(0.075) + ConvFuser + TransFusionHead(out_size_factor 8). 원 BEVFusion(ResNet-50, stride 8)과의 차이는 README에 명시. 전 조합 공통: nuScenes 6-camera, **단일 sweep**(`lidar_sweeps: 0`, LoadPointsFromMultiSweeps 제거), GT-Aug 미사용, CBGS, image_size [256,704], 동일 augmentation, 동일 optimizer/schedule(현재 신규 config의 것), max_epochs 20.
- **D3 config 계열** — 디렉토리 `configs/nuscenes/det/ablation/`:
  - `default.yaml`: B0 전체 모델 블록 + 공통 파이프라인/optimizer. `type: ModularBEVFusion`.
  - leaf 8개(각 파일은 바뀌는 블록만 override, 단 head cfg는 lidar encoder에 맞춰 함께 override):
    | 파일 | LiDAR | Camera vtransform | Fuser | Head |
    |---|---|---|---|---|
    | `b0_legacy.yaml` | Sparse | DepthLSS | Conv | TransFusion |
    | `e1_dsvt.yaml` | **DSVT** | DepthLSS | Conv | TransFusion(0.3/2) |
    | `e2_widthformer.yaml` | Sparse | **WidthFormer** | Conv | TransFusion |
    | `e3_gfusion.yaml` | Sparse | DepthLSS | **DGF** | TransFusion |
    | `e4_dal.yaml` | Sparse | DepthLSS | Conv | **DAL** |
    | `c2_dsvt_widthformer.yaml` | DSVT | WidthFormer | Conv | TransFusion(0.3/2) |
    | `c3_dsvt_widthformer_gfusion.yaml` | DSVT | WidthFormer | DGF | TransFusion(0.3/2) |
    | `c4_full.yaml` | DSVT | WidthFormer | DGF | DAL(0.3/2) |
  - `c4_full.yaml`의 resolved 모델 블록은 현재 기본 config `dsvt_dgf_dal_widthformer_0p3.yaml`과 **동일해야 함**(테스트 T2). 기존 config 파일들은 삭제·수정하지 않는다.
  - DSVT 초기화: leaf에 `load_from`을 넣지 않는다. README에 `--load_from pretrained/dsvt_nuscenes_official_lidar.pth` 명령을 조합별로 기재(DSVT 포함 조합만).
- **D4 생성기** — `tools/ablation/gen_configs.py`: 모듈 블록 4축(lidar/vtransform/fuser/head)을 python dict로 정의하고 8개 leaf yaml을 생성. 커밋된 yaml과 생성 결과가 같아야 함(T1). 생성기는 yaml 주석 헤더(생성 시각 제외, 조합 설명)를 포함.
- **D5 서버 smoke** — `tools/ablation/smoke_all.sh`: 8개 config 각각 `tools/smoke_train.py`(또는 동등)로 1-step forward/backward를 순차 실행하고 결과 표를 `tools/ablation/smoke_results.md`에 append. 로컬(mmcv 없음)에서는 실행 불가 → 서버용. 스크립트 문법만 `bash -n`으로 확인.
- **D6 실험 원장** — `experiments/LEDGER.md` 템플릿: 열 = 실험ID, config, git commit, 초기화(DSVT 공식/ImageNet/scratch), 데이터(full / 1/4 subset), sweeps, epochs, best epoch, mAP, NDS, 클래스별 AP 링크, Thor/Orin Engine A/B/C·E2E P50, 비고. 8행을 미리 채워 두되 수치는 비움.

## 3. DoD (로컬 CPU에서 가능한 것만; mmcv·torchpack 없음)
- T1 `python tools/ablation/gen_configs.py --check` → 커밋된 8개 yaml과 byte-동일.
- T2 순수 yaml 병합 테스트(torchpack 없이 `tools/ablation/tests/test_configs.py`에서 default.yaml + leaf를 깊은 병합해 모델 블록 비교): `c4_full` 병합 결과의 `model.encoders/fuser/heads/decoder` == 기존 기본 config 병합 결과. `${...}` 표현식은 문자열 그대로 비교.
- T3 `ModularBEVFusion` 단위 테스트: mmcv 없이 실행 가능한 범위로 fuser/head 디스패치 로직을 함수로 분리(`select_fuser_call`, `select_head_call`)하고 fake 모듈로 4조합 × (fusion/lidar-only/camera-only) 호출 경로를 검증. 모델 본체 import는 mmcv 필요 → 서버 smoke로 넘김.
- T4 `bash -n tools/ablation/smoke_all.sh`.
- T5 문서: `tools/ablation/README.md`(계열 표, 실행 명령, B0 정의와 원 BEVFusion 차이, DSVT 초기화 명령), `new-arch.md`에 절 추가.
- git: staging까지만(커밋은 오케스트레이터). `tools/ablation/EXEC_phase2.md`에 기록.

## 4. 금지
- 기존 config 파일·`bevfusion.py`·`dynamic_bevfusion.py`의 동작 변경 금지(alias 전환은 동작 동일 조건에서만). deployment 코드 수정 금지.
- 서버 없이 검증 못 한 것을 "검증됨"으로 쓰지 않는다.

## 5. 추가 결정 (2026-09-17 밤, 사용자): 전체 조합 16개
- 4개 축(DSVT, WidthFormer, GFusion, DAL)의 **2⁴ = 16개 조합 전부**를 config로 생성한다. 파일명은 4비트 체계 `dsvt{0|1}_wf{0|1}_gf{0|1}_dal{0|1}.yaml` (예: `dsvt1_wf0_gf1_dal0.yaml` = DSVT + GFusion만).
- §2 D3의 8개 이름(b0_legacy, e1~e4, c2~c4)은 **별칭 파일**(동일 내용, 생성기가 함께 생성)로 유지한다. 별칭 표를 README에 둔다.
- head cfg 종속 규칙(DSVT면 0.3/2, 아니면 0.075/8)은 생성기가 자동 적용한다.
- 원장(`experiments/LEDGER.md`)은 16행이며 "우선 실행" 열로 1차 9개(b0, e1~e4, c2~c4)를 표시한다.
