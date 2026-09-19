# SPEC — ModularBEVFusion(HEAD) vs DynamicBEVFusion(829c7ae) 수치 동치 (CPU, FINAL config)

## 동기 (2026-09-19)
B200 스크리닝에서 정확도가 비정상(epoch 1 NDS ≤ 0.07). 완전 해석 config 비교 결과 레거시 성공 config와의 차이는 `model.type`
(DynamicBEVFusion → ModularBEVFusion)과 DSVT 옵션 2개뿐. 래퍼가 손실/출력을 바꾸는지 수치로 확인한다.

## 요구
1. `git worktree add /tmp/bev_legacy 829c7ae`(읽기 전용 참조). 도커 이미지 `bevfusion-train:cu113`(torch 1.10 CPU, mmcv 1.4, torchpack)에서 실행.
   마운트 패턴은 `tools/ablation/run_smoke_local.sh` 참고(ops .so는 이미지 내부 것 사용 → `mmdet3d/ops`는 마운트 금지; `mmdet3d/models`, `mmdet3d/core`, `mmdet3d/datasets`, `configs`, `tools`만 마운트). GPU 불필요(`--gpus` 옵션 없이).
2. 모델: HEAD의 `configs/nuscenes/det/ablation/dsvt1_wf1_gf1_dal1.yaml`(FINAL, ModularBEVFusion)과 829c7ae의
   `configs/nuscenes/det/transfusion/secfpn/lidar/dsvt_dgf_dal_widthformer_0p3.yaml`(DynamicBEVFusion)을 각각 해당 코드 트리로 빌드
   (컨테이너 두 번 실행: 한 번은 HEAD 트리 마운트, 한 번은 /tmp/bev_legacy 트리 마운트). 같은 seed로 초기화한 뒤 **state_dict를 파일로 저장해 서로 로드**(키 집합·shape 차이는 보고, 공통 키만 로드).
3. 입력: 합성 배치 1개(포인트 N=20000 5차원 범위 내 랜덤, 이미지 6×3×256×704 랜덤, 카메라 행렬은 nuScenes mini의 실제 샘플 1개에서 가져오기 — `/mnt/hdd/nuscenes` mini + 파이프라인으로 `dataset[0]`을 뽑아 pickle로 저장 후 양쪽에서 동일하게 사용하는 방식 권장). GT 박스/라벨도 같은 샘플 것.
4. 비교: (a) eval 모드 `forward(return_loss=False)` 출력 boxes_3d tensor·scores — 정렬 후 max abs diff; (b) train 모드 손실 딕셔너리 각 항목 값 — 상대 오차. 중간: fuser 입력(카메라 BEV, LiDAR BEV) 텐서도 hook으로 저장해 비교.
5. 판정: 손실 항목·출력이 1e-4 상대 오차 안이면 PASS. 아니면 처음 어긋나는 단계와 코드 위치. mmdet3d/ 수정 금지(진단만).
6. 산출물: `tools/ablation/wrapper_equiv_check.sh`(재실행), `tools/ablation/wrapper_equiv_report.json`, `tools/ablation/EXEC_wrapper_equiv.md`.
