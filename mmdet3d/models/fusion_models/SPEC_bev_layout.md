# SPEC — BEV 축 규약 통일 (`bev_layout`, 표준 = [B, C, Y, X])

## 원인 (2026-09-20 확정)
- 레거시 BEVFusion 모듈은 BEV를 **[B, C, X, Y]** 로 다룬다: `SparseEncoder` dense 출력, `BaseTransform`(DepthLSS/LSS) `bev_pool` 출력,
  `TransFusionHead`(`create_2D_grid`가 [X,Y] 평탄화, heatmap 타깃을 `center_int[[1,0]]`로 row=x, col=y에 그림).
- 새 모듈은 **[B, C, Y, X]**: `DSVTLidarEncoder`(공식 OpenPCDet 규약), `WidthFormerTransform`(docstring "canonical [B,C,Y,X]"),
  `DALDecoupledHead`(자체 `create_2D_grid`·`_dense_heatmap_target`로 [Y,X] 타깃), `DepthGFusion`(`camera_layout`/`lidar_layout` 기본 "yx").
- `ModularBEVFusion`은 규약을 검사·변환하지 않아 조합에 따라 특징과 타깃이 전치됨.
  B200 실측: B0/A4(xy+xy) 정상, FINAL(yx+yx) 0.488, A1/C1/P2(DSVT yx + TransFusion xy) 붕괴, A3(레거시 xy + DAL yx) 붕괴 예측.

## 요구
1. **표준 규약 `CANONICAL_BEV_LAYOUT = "yx"`** ([B, C, Y, X], row = y). 모듈 클래스 속성 `bev_layout: str`:
   - `"xy"`: `SparseEncoder`, `BaseTransform` 계열(DepthLSS/LSS/aware_bevdepth 등 `bev_pool` 사용), `PointPillars` scatter 등 레거시 LiDAR 인코더.
   - `"yx"`: `DSVTLidarEncoder`, `WidthFormerTransform`.
   - 속성이 없는 인코더/vtransform은 **에러**(암묵적 규약 금지).
2. `ModularBEVFusion.forward_single`: 각 센서 BEV를 얻은 직후 해당 모듈의 `bev_layout`이 표준과 다르면 `transpose(-1, -2).contiguous()`.
   (`camera_bev` 외부 주입은 표준으로 간주.) 퓨저·디코더는 표준만 본다. `DepthGFusion`은 `camera_layout`/`lidar_layout`을 "yx"로 받게 되므로 기본값 그대로.
3. `TransFusionHead`에 `bev_layout` 인자(기본 `"xy"` = 레거시 `BEVFusion` 클래스 호환). `"yx"`일 때 `create_2D_grid`와 dense heatmap 타깃을 [Y,X]로 생성
   (현재 `DALDecoupledHead`의 구현을 부모로 옮기고 DAL은 그것을 상속·재사용). `ModularBEVFusion`은 heads를 빌드할 때 `bev_layout="yx"`를 주입한다.
   `DALDecoupledHead`는 동작 불변(현재 이미 yx).
4. 레거시 `BEVFusion`/`DynamicBEVFusion` 클래스 경로는 **바꾸지 않는다**(기존 xy 체인 유지). `DynamicBEVFusion`이 `ModularBEVFusion`의 별칭이면 별칭 그대로.
5. 테스트(CPU, mmdet3d import 없이 가능한 부분 + 도커 CPU로 가능한 부분):
   - (a) 16개 ablation config를 해석해 인코더·vtransform·헤드 클래스의 `bev_layout` 선언 존재 확인, 헤드 주입값 "yx" 확인.
   - (b) 헤드 왕복: 합성 GT 박스 (x=a, y=b) → `bev_layout="yx"` 헤드의 heatmap 타깃 최대값 위치가 `[row=b_idx, col=a_idx]`; `create_2D_grid`의 평탄 인덱스 `row*W+col` 위치 값이 `(a_idx+0.5, b_idx+0.5)`. `"xy"`(레거시)는 반대임을 함께 고정.
   - (c) 전치 경로: `bev_layout="xy"` 가짜 인코더 출력 [B,C,X,Y]가 forward_single 안에서 [B,C,Y,X]로 바뀌는지(hook 또는 가짜 fuser로 확인).
   - (d) DSVT+TransFusion(A1 config) 합성 배치 손실 계산이 예외 없이 되는지(도커 CPU; spconv 불필요).
6. 문서: `mmdet3d/models/fusion_models/README_bev_layout.md`에 규약·모듈별 선언·전치 위치를 표로. `experiments/HISTORY_20260919.md` §2에 원인 항목 추가.
7. 산출물: 코드, 테스트(`tests/test_bev_layout.py`), README, `EXEC_bev_layout.md`. 커밋 금지.
