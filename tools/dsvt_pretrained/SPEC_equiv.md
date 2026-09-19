# SPEC — DSVT 공식 구현 대비 수치 동치 검증 (Phase 1c)

## 동기
B200 스크리닝에서 DSVT 사전학습을 쓴 A1이 사전학습 없는 B0보다 못 배움(epoch 1: matched IoU 0.10 vs 0.20, mAP 0.046).
Phase 1의 "이식 검증"은 state_dict 키·shape 일치(340/345)까지였고 **공식 구현과 출력이 수치로 같은지는 확인하지 않았다.**
가중치가 구조는 맞지만 의미가 다른 위치(set partition 순서, 윈도 시프트, 위치 임베딩 규약 등)에 들어갔으면
DSVT 출력은 무의미해지고 모델은 카메라에만 의존한다 — 관측과 일치하는 가설.

## 요구
1. 공식 DSVT 저장소(Haiyang-W/DSVT, Apache-2.0)가 `/tmp/DSVT`에 clone되어 있다(네트워크 불필요).
   nuScenes 설정 `/tmp/DSVT/tools/cfgs/dsvt_models/dsvt_plain_1f_onestage_nusences.yaml`의
   VFE + backbone_3d(DSVT) + map_to_bev + backbone_2d(BaseBEVResBackbone) 부분만
   **순수 PyTorch CPU로 실행 가능한 형태로 vendoring**한다: `tools/dsvt_pretrained/official_ref/` 아래에 필요한 파일만 복사하고,
   pcdet 의존(레지스트리, cfg 객체, CUDA op)은 최소 shim으로 대체. `ingroup_inds` CUDA op
   (`/tmp/DSVT/pcdet/ops/ingroup_inds/`)은 동일 의미의 순수 torch CPU 구현으로 대체(그룹 id별 등장 순서 인덱스).
   코드 위치: `/tmp/DSVT/pcdet/models/backbones_3d/{dsvt.py,dsvt_input_layer.py}`, `/tmp/DSVT/pcdet/models/backbones_3d/vfe/`,
   `/tmp/DSVT/pcdet/models/backbones_2d/`. 라이선스 헤더/출처 명시.
2. **포인트 → 공식 BEV feature(backbone_2d 출력 `spatial_features_2d`)** 경로를 공식 코드로 재현하고,
   `pretrained/DSVT_Nuscenes_val.pth`를 로드한다(키 매핑 없이 원본 그대로, strict).
3. 우리 경로: `mmdet3d/models/backbones/dsvt_core.py`의 `DSVTLidarEncoder`(official_layout=True, zero_feature_channels=[4],
   학습 config와 같은 인자 — `configs/nuscenes/det/ablation/default.yaml`의 lidar DSVT 블록 참조)에
   `pretrained/dsvt_nuscenes_official_lidar.pth`(prefix `encoders.lidar.backbone.` 제거)를 로드한다.
   mmdet3d 패키지 import 없이 `dsvt_core.py`를 파일 경로로 직접 import(순수 torch, torch_scatter 없음 → 내장 대체 경로 사용).
4. 입력: `/home/hanelso/data/nuscenes/samples/LIDAR_TOP/*.pcd.bin` 중 2개(x,y,z,intensity,ring; 공식은 5번째 = timestamp 0, 우리는 ring을 0으로).
   동일 포인트를 두 경로에 넣고 (a) voxelization 결과(pillar 좌표 집합·개수 일치 여부), (b) VFE 출력(pillar feature),
   (c) DSVT 블록별 출력, (d) BEV feature(공식 spatial_features / 우리 backbone 출력 직전), (e) backbone_2d 출력까지
   단계별 max abs diff, rel diff, cosine을 표로 기록. eval 모드, float32.
   pillar 순서가 다를 수 있으므로 좌표 키로 정렬해 비교.
5. 판정: 단계별 diff가 1e-3 이하(허용 오차 명시)이면 PASS; 아니면 **처음 어긋나는 단계**와 원인 후보(코드 위치, 어떤 규약 차이인지)를 EXEC에 적는다.
   mmdet3d/ 코드는 수정하지 말고 진단만.
6. 산출물: `tools/dsvt_pretrained/equiv_check.py`(재실행 가능), `tools/dsvt_pretrained/equiv_report.json`, `tools/dsvt_pretrained/EXEC_equiv.md`.
   호스트 python(torch 1.12 CPU)으로 돌 것. 패키지 설치 금지.
