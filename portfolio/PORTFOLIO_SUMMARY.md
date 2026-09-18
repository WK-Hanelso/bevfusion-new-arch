# BEVFusion New Architecture — Additional Work

## One-line

DSVT, WidthFormer, depth-guided fusion, DAL을 하나의 `[B,C,Y,X]` BEV 규약으로 통합하고, mAP가 0에 고정된 DAL heatmap target 전치 결함을 격리·수정해 문서화된 최고 mAP 0.5281 / NDS 0.5132를 확보했다.

## Problem

Camera와 LiDAR 분기를 한 BEV 공간에서 결합하면서 feature layout과 target 좌표 규약이 서로 달랐다. Loss는 감소했지만 validation mAP는 0에 고정되어 단순 최적화 실패가 아닌 target 생성 경로를 의심해야 했다. 동시에 8-GPU full training 결과와 배포 진단값을 과장 없이 분리해 설명할 evidence가 필요했다.

## Decision

- 모든 BEV tensor의 공개 규약을 `[B,C,Y,X]`로 고정했다.
- 분류는 fused feature, box regression은 LiDAR-only feature를 사용하는 DAL 구조를 유지했다.
- 기존 TransFusion 경로를 수정하지 않고 DAL head에서 heatmap target만 `[Y,X]`로 재생성했다.
- Single-sweep은 latency 목표를 위한 의도된 입력 결정으로 유지하고, velocity accuracy 비용을 별도 지표로 명시했다.

## What I Built

6-camera WidthFormer branch와 DSVT LiDAR branch를 DepthGFusion으로 결합하고 DAL head로 전달하는 `DynamicBEVFusion` 경로를 구성했다. 좌표 변환은 공통 BEV grid utility에서 검증하며, target fix는 DAL head 내부에 국소화했다. 학습과 배포 계약은 별도로 유지했다.

## Evidence

- Before fix: validation mAP **0.0000** — 로컬 저장소 문서 기록, raw log 미보존.
- Target alignment: **43.2% (725/1,680) → 86.0% (1,444/1,680)** — 저자 공개 페이지 기록.
- One epoch after fix: **mAP 0.2045** — 저자 공개 페이지 기록이며 다른 mAP 지점과 학습 구간이 다름.
- Full training: **8 × RTX A6000 48GB**, FP32, global batch 32, 20 epochs.
- Best epoch 19: **mAP 0.5281 / NDS 0.5132**.
- Final epoch 20: **mAP 0.5258 / NDS 0.5129**.
- Single-sweep vs multi-sweep reference: **mAVE 1.132 vs 0.257**, NDS velocity term **0.000 vs 0.743** — 비교값은 저자 공개 페이지 기록.
- Precision record: FP16은 cyclic LR 9.2e-4 부근에서 NaN, FP32 full run은 20 epochs 완주 — 로컬 문서 기록. Epoch 2와 시간·메모리 상세는 저자 공개 페이지 기록.
- Deployment diagnostic: parallel P50 **26.4405 ms** — legacy layout, random-init engine, synthetic input 한정.

## Failure / Debugging

DAL prediction은 `[Y,X]` BEV를 사용했지만 상속 target 경로는 legacy `[X,Y]` 순서로 gaussian을 배치했다. 이 전치 불일치로 target과 prediction이 공간적으로 어긋났고 mAP가 0에 고정됐다. `get_targets_single`에서 dense heatmap만 `[Y,X]`로 다시 생성하도록 수정하고, 다른 target과 기존 TransFusion 경로는 보존했다.

저자가 공개한 학습 기록에서는 val-frame GT box center와 LiDAR occupancy의 일치가 수정 전 43.2%에서 수정 후 86.0%로 증가했고, fix 직후 1 epoch mAP 0.2045가 기록됐다. 이 수치의 원시 학습 로그와 공식 nuScenes 평가 출력은 이 머신에 없으므로 독립 재현 가능한 결과로 표기하지 않는다.

FP16은 cyclic LR peak 부근 NaN으로 중단되어 FP32로 full schedule을 완료했다는 사실은 로컬 학습 가이드에 있다. 저자 공개 기록상 FP16은 epoch 2에서 중단됐고 0.60 s/iter·12 GB, FP32는 0.86 s/iter·19 GB였지만 raw log는 현재 저장소에 없다.

## Result

저장소 문서 기준 최고 성능은 epoch 19 mAP 0.5281 / NDS 0.5132다. Single-sweep mAVE 1.132는 로컬 문서에도 기록되어 있으며, 저자 공개 페이지의 동일 codebase·dataset·evaluation multi-sweep reference는 mAVE 0.257이다. NDS velocity term 0.000 대 0.743은 latency를 위해 선택한 single-sweep의 비용이 velocity 항에서 약 0.07 NDS였음을 보여준다. 이는 의도된 trade-off이며 모델 결함으로 해석하지 않는다.

## Scope / Limitation

이 package에서는 새 학습·추론을 실행하지 않았다. Full-training raw log, 공식 nuScenes evaluation output, 선택 checkpoint는 현재 머신에 없다. Alignment 전후 값, fix 후 1-epoch mAP, multi-sweep reference, FP16/FP32 상세 표는 `portfolio/sources/notion_page_excerpt.md`에 담긴 저자 공개 기록이며 모두 non-reproducible로 분류했다. 로컬 파일에도 있는 최고 mAP/NDS와 single-sweep mAVE는 로컬 문서를 primary source로 유지했다. Deployment latency는 학습 weight나 실제 센서 입력의 성능을 뜻하지 않는다.

## Portfolio Assets

- `assets/01_architecture.svg` / `.png`: 두 센서 분기, depth-guided fusion, DAL, 공통 BEV 좌표 규약.
- `assets/02_target_bug_recovery.svg` / `.png`: target alignment before/after와 서로 다른 세 학습 구간의 mAP.
- `assets/03_single_sweep_tradeoff.svg` / `.png`: single-sweep latency 결정에 따른 velocity metric 비용.
- `assets/04_fp16_vs_fp32.svg`: precision별 안정성, iteration time, GPU memory 표.

상세 Source of Truth는 `EVIDENCE.md`와 `evidence.json`에 있다.
