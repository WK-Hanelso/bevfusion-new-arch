# SPEC — 검증 매트릭스 (긴 실행 전 단 한 번, 전부 통과해야 실행)

## 동기 (2026-09-19)
검증이 부분 집합·사후 대응으로 이뤄져 이틀을 잃음. 사용자: "검증을 똑바로 안 하고 간헐적으로만 하니까 이런 것." 이 매트릭스는 긴 실행 전에 **한 번에 전부** 도는 검증이며, 여기서 PASS한 커밋만 스크리닝/최종을 띄운다.

## 항목 (모두 실제 GPU, conda env bevfusion-b200, 실제 nuScenes; 총 8 GPU에서 약 1시간)
| # | 항목 | 방법 | 판정 |
|---|---|---|---|
| M1 | 16개 config 전부 학습 forward/backward + ckpt + 평가 | 512샘플 서브셋, **2 epoch**(ckpt 회전·두 번째 평가 포함), GPU 1장, samples_per_gpu 16 + cumulative 2 | 각 config: epoch_2.pth 존재, 평가 2회 `mAP:` 출력, 손실 유한 |
| M2 | resume | M1의 epoch_1.pth에서 `--resume_from`으로 1 epoch 더 | 로그에 `resumed epoch 1` 상당, epoch 2 완료 |
| M3 | 4 GPU DDP(최종 단계 구성) | B0·FINAL을 4 GPU, samples_per_gpu 8로 1 epoch(서브셋) + 평가 | 완료·평가 출력 |
| M4 | 2 GPU DDP(스크리닝 대안 구성) | B0·FINAL 2 GPU, samples_per_gpu 16, 1 epoch | 완료 |
| M5 | 평가 예외 | 랜덤 초기화 모델로 test.py 평가(박스 0개 가능) | 예외 없이 `mAP:` 출력 |
| M6 | 런처 실패 경로 | 가짜 실패 명령으로 재시도·큐 진행(기존 유닛 테스트) | pytest PASS |
| M7 | 집계 | M1 로그 16개로 aggregate.py 실행 | CSV/MD 생성, TOP_K_IDS 출력 |
| M8 | (선택) spconv v2 | 레거시 spconv vs spconv v2 SparseEncoder 출력 동치(같은 가중치·입력) | max abs diff ≤ 1e-3, 배치 32/GPU 학습 1 epoch 완료 |

## 산출물
- `tools/ablation/verify_matrix.sh <nuscenes_root> [--with-spconv2]`: 위 항목을 8 GPU에 병렬 배치해 실행, 화면에 60초마다 상태, 끝나면 표와 `experiments/verify/PASS_<rev>.txt`(전부 PASS일 때만).
- `tools/ablation/run_screening.sh`/최종 실행기는 이 마커(또는 기존 게이트 마커)를 요구.
- `EXEC_verify_matrix.md`: 결과 표.
