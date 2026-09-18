# Source: Notion portfolio page "04. BEVFusion New Arch — 멀티모달 모델 학습과 Failure Analysis" (NAVER portfolio, last edited 2026-09-18). Excerpted 2026-09-19 by orchestrator.
# Numbers below are the author's published statements. Where a local repo file (new-arch.md, tools/README.md, deployment/runtime/evidence/*.json) also contains the value, cite the local file as primary. Values that exist ONLY here: cite this excerpt, Reproducible = no.

## Training (nuScenes v1.0-trainval, train 28,130 / val 6,019)
- 8 GPU DDP, global batch 32, 20 epoch, ~23 h
- single-sweep LiDAR, no GT-Aug, camera backbone ImageNet init only
- best epoch 19: mAP 0.5281 / NDS 0.5132
- class AP: car 0.806 / pedestrian 0.767 / bus 0.630 / construction_vehicle 0.165 / bicycle 0.255 / trailer 0.385

## Heatmap target transpose defect
- First runs: mini 20 epoch and full 2 epoch, mAP exactly 0.0000, heatmap loss frozen at background floor
- Cause: new head redefined BEV grid y-major [Y, X], inherited target builder stamped gaussians in legacy [X, Y] order -> prediction/target transposed
- Verification: GT box centers vs LiDAR occupancy grid on val frames. Before fix batch: 43.2% agreement. After fix: 86.0%.
- Fix scoped to new head's target builder only (legacy path untouched). 1 epoch right after fix: mAP 0.2045.

## FP16 instability
- Config default FP16 -> NaN near cyclic LR peak, Hungarian assigner crash; full schedule completed in FP32.

## Single-sweep cost (same evaluation)
- mAVE 1.132 m/s (single-sweep, this model) vs multi-sweep model mAVE 0.257; velocity term alone ~0.07 NDS difference. Structural consequence of the single-sweep latency decision, not a model defect.

## Thor deployment (FP16, synthetic 34,688 points, 100 iterations)
- Camera engine 4.99 ms / LiDAR engine 12.60 ms / Fusion engine 8.79 ms (each alone)
- Full parallel P50 / P99: 26.44 / 27.31 ms; serial 26.38 ms -> no speedup despite ~10.3 ms overlap (Nsight)
- 5 CUDA plugins; bundle = 3 engines + manifest + plugins SHA-256; 18 contract tests
- Latency measured with random-init engines; accuracy of trained-weight FP16 engines not yet verified

# Source 2: Notion subpage "11-1. 학습 기록 — heatmap 결함 규명, FP16 불안정, 공식 DSVT 대비 격차 분석" (last edited 2026-09-17). Author states the numbers were copied from training logs and official nuScenes eval output; those logs are NOT on this machine.

## Run facts
- 8 × RTX A6000 48GB DDP; 23 h 03 min for 20 epochs; batch 32 (8×4); 3,862 iter/epoch; FP32; sweeps 0; ResNet-34 ImageNet camera backbone; DSVT-Pillar voxel 0.3, d_model 128, set 90, block 4, window 30×30×1; CBGS, no GT-Aug; 19.7 GB/GPU, 0.93 s/iter
- best epoch 19 mAP 0.5281 / NDS 0.5132; epoch 20 0.5258 / 0.5129
- TP metrics epoch 19: mATE 0.3176 m, mASE 0.2569, mAOE 0.6425 rad, mAVE 1.1320 m/s (NDS contribution 0.000), mAAE 0.2913. NDS = (5×0.5281 + 2.492)/10 = 0.5132 checks.

## Heatmap target transpose defect (detail)
- Symptom: nuScenes-mini 20 epoch and full 2 epoch, mAP exactly 0.0000; heatmap loss frozen at 3.20; matched IoU ~0.01
- Cause: draw_heatmap_gaussian expects center[0]=col, center[1]=row. Legacy TransFusionHead flattens grid x-major [X,Y] and passes center_int[[1,0]] (correct for legacy). New DALDecoupledHead flattens y-major [Y,X] but inherited the legacy target builder -> prediction [Y,X], target [X,Y].
- Measured evidence (val frames, GT box centers vs LiDAR occupancy grid): before fix (row=x, col=y) 725 / 1,680 = 43.2%; after fix (row=y, col=x) 1,444 / 1,680 = 86.0%
- Same-interval comparison: before fix heatmap loss moved 0.07 over 750 iter; after fix dropped 1.09. mAP 0.2045 after 1 epoch post-fix.
- Fix: target builder overridden in the new head only; legacy path untouched.

## FP16 vs FP32
| FP16 | NaN death at epoch 2, LR 9.2e-4 (Hungarian assigner: matrix contains invalid numeric entries) | 0.60 s/iter | 12 GB |
| FP32 | 20 epochs completed incl. LR peak 1e-3 | 0.86 s/iter | 19 GB |

## Single-sweep cost (same dataset, same evaluation, same codebase)
| this run | sweeps 1 | mAVE 1.132 | NDS velocity term 0.000 |
| multi-sweep BEVFusion (same codebase) | multi | mAVE 0.257 | NDS velocity term 0.743 |
-> ~0.07 NDS from the velocity term alone.

## Gap vs official DSVT nuScenes (LiDAR-only): official mAP 66.4 / NDS 71.1 vs this 52.8 / 51.3, identical hyperparameters; attributed to single sweep, no GT-Aug, recipe. Orientation error bus 0.636 / trailer 0.806 flagged for investigation.
