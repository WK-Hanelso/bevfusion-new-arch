# Portfolio Asset Captions

## 01_architecture.svg / 01_architecture.png

Notion 위치: Additional Work 본문 첫 문단 직후, 문제 정의와 구현 설명 사이.

Caption: 6-camera WidthFormer와 DSVT LiDAR branch를 depth-guided fusion으로 결합하고 DAL head에서 분류와 회귀를 분리했다. 전체 경로는 `[B,C,Y,X]` BEV 좌표 규약을 공유한다.

Claim boundary: 구현 구조와 tensor layout만 보여준다. 정확도 향상 폭, 실시간성, production 적용을 주장하지 않는다. 근거는 C01–C02.

## 02_target_bug_recovery.svg / 02_target_bug_recovery.png

Notion 위치: Failure / Debugging 섹션의 첫 그림.

Caption: DAL target의 `[X,Y]` 전치 결함을 `[Y,X]`로 수정한 뒤, val-frame GT center와 LiDAR occupancy의 일치율이 43.2%(725/1,680)에서 86.0%(1,444/1,680)로 회복됐다. 서로 다른 학습 구간에서 mAP는 0.0000에서 fix 직후 1 epoch 0.2045, best epoch 19 0.5281(NDS 0.5132)로 변했다.

Claim boundary: 최고 mAP/NDS와 결함 원인은 로컬 문서·코드에도 기록되어 있다. Alignment, fix 직후 1-epoch mAP, before-fix 학습 구간은 저자가 공개한 페이지의 기록이며 raw training log는 이 머신에 없다. 세 mAP 값은 하나의 연속 curve가 아니라 각각 다른 학습 구간이다. 근거는 C04–C09, C12, C14.

## 03_single_sweep_tradeoff.svg / 03_single_sweep_tradeoff.png

Notion 위치: Result 다음, latency 목표와 accuracy trade-off를 설명하는 문단.

Caption: 동일 codebase·dataset·evaluation에서 single-sweep run의 mAVE는 1.132, multi-sweep BEVFusion reference는 0.257이었다. NDS velocity term은 각각 0.000과 0.743으로, single-sweep latency 결정을 위해 velocity 항에서 약 0.07 NDS 비용이 측정됐다.

Claim boundary: Single-sweep mAVE 1.132와 velocity term 0.000은 로컬 문서에도 있다. Multi-sweep mAVE·velocity term과 약 0.07 NDS 해석은 저자가 공개한 페이지의 기록이며 raw evaluation output은 이 머신에 없다. End-to-end latency 개선이나 전체 NDS 차이를 주장하지 않는다. 근거는 C11, C15–C16.

## 04_fp16_vs_fp32.svg

Notion 위치: Failure / Debugging 섹션의 precision 안정성 문단 옆 선택 표.

Caption: FP16은 epoch 2의 cyclic LR 9.2e-4 부근에서 NaN으로 중단됐고, FP32는 LR peak 1e-3을 포함한 20 epochs를 완주했다. 저자 기록상 FP16/FP32는 각각 0.60/0.86 s/iter, 12/19 GB였다.

Claim boundary: FP16 NaN과 LR, FP32 full-run completion은 로컬 학습 가이드가 primary source다. Epoch·iteration time·memory 표는 저자가 공개한 페이지에서만 확인된다. Raw training log는 이 머신에 없으며 이 portfolio 작업에서 새 실험을 수행하지 않았다. 근거는 C17–C18.
