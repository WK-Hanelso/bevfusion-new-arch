# 16조합 full 학습 일정 — B200 8장 (작성 2026-09-18)

## 0. 전제·산정 근거
- 기준 실측: c4_full(legacy 구조) nuScenes full 20 epoch = **8×A6000, batch 32(8×4), 23h 3분**, GPU당 19.7GB, 0.93 s/iter, 3,862 iter/epoch.
- B200: 메모리 180GB(GPU당 batch 8~16 가능), FP32/TF32 처리량은 A6000의 **약 3~5배**로 가정(sparse·scatter·데이터 로딩 비중이 커서 이론치보다 낮게 잡음). 실제 배수는 Day 0 mini 학습에서 s/iter로 확정한다.
- 단일 실험 = 4×B200, batch 32(4×8) 유지(레시피 동일성) → 예상 **8~12h/run**. 동시 2개 실험.
- 총 16 run → 8 wave × 8~12h = **64~96h(2.7~4일)**. 데이터 로딩이 병목이면 wave당 +20% 여유.
- 정밀도: **FP32(TF32 허용)**. FP16 학습은 NaN 이력으로 제외. BF16은 비교 가능성 훼손이라 본 사다리에서 제외(별도 실험 후보).
- 초기화 정책: `dsvt1_*` 8개는 **공식 DSVT ckpt 초기화**(기본 구조 정책), 카메라 backbone ImageNet, 나머지 scratch. 초기화 효과 분리를 위해 `dsvt1_wf1_gf1_dal1` **scratch 1회 추가**(run 17).
- 체크포인트 보관: best(NDS) + last만(각 ~0.5GB) → 17 run ≈ 17GB. 매 epoch 평가는 유지(곡선 확보).

## 1. 일정
| Day | 작업 | GPU | 산출물 |
|---|---|---|---|
| D0 (접속일) | 접속·인벤토리(nvcc/driver/CPU 코어/디스크/데이터) → conda 환경 구축(`setup_b200_conda.sh`, 1~2h) → nuScenes full infos 생성(27분, 이미 있으면 생략) → 16 config smoke(1-step) → **mini 3 epoch 학습 1회**로 s/iter·메모리 실측 → 일정 배수 확정 | 1~4 | `smoke_results.md`, `LEDGER.md` 환경 행 |
| D1 | Wave 1: `dsvt0_wf0_gf0_dal0`(B0) ‖ `dsvt1_wf1_gf1_dal1`(c4_full, DSVT init) | 4+4 | 기준선 2개 |
| D1~2 | Wave 2: `dsvt1_wf0_gf0_dal0`(e1) ‖ `dsvt1_wf1_gf1_dal0`(c3) | 4+4 | |
| D2 | Wave 3: `dsvt0_wf1_gf0_dal0`(e2) ‖ `dsvt0_wf0_gf1_dal0`(e3) | 4+4 | 단독 4개 완성 |
| D2~3 | Wave 4: `dsvt0_wf0_gf0_dal1`(e4) ‖ `dsvt1_wf1_gf0_dal0`(c2) | 4+4 | 1차 9개 완성 |
| D3 | Wave 5: `dsvt1_wf0_gf1_dal0` ‖ `dsvt1_wf0_gf0_dal1` | 4+4 | 둘씩 조합 |
| D3~4 | Wave 6: `dsvt0_wf1_gf1_dal0` ‖ `dsvt0_wf1_gf0_dal1` | 4+4 | |
| D4 | Wave 7: `dsvt0_wf0_gf1_dal1` ‖ `dsvt1_wf1_gf0_dal1` | 4+4 | |
| D4~5 | Wave 8: `dsvt1_wf0_gf1_dal1` ‖ `dsvt0_wf1_gf1_dal1` | 4+4 | 16개 완성 |
| D5 | Wave 9: `dsvt1_wf1_gf1_dal1` **scratch** ‖ (여유: 실패 재실행 또는 10-sweep 표준 조건 1회) | 4+4 | run 17(+18) |
| D5~6 | 집계: LEDGER 16(+2)행 mAP/NDS/클래스별, epoch 곡선, best ckpt 수집 → ONNX export(c4_full 등 선택) → Orin/Thor latency | 0~1 | 논문용 표 v1 |

wave 순서는 **결론에 중요한 것부터**(기준선·full → 단독 → 누적 → 나머지 쌍). 앞 wave가 예상보다 빠르면 다음 wave를 즉시 당긴다.

## 2. 운영 규칙
- 실행 단위: `CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 tools/train_torchrun.py <config> --run-dir runs/<실험ID>_<config> --find_unused_parameters True --fp16 None [--load_from pretrained/dsvt_nuscenes_official_lidar.pth]`.
- 각 run 시작 시 LEDGER에 실험ID·config·commit·초기화·시작시각 기록, 종료 시 best epoch·mAP·NDS 기록. 실패 시 원인·재실행 여부 기록.
- 감시: 1 epoch마다 loss/NaN/GPU 메모리 확인. NaN 발생 시 즉시 중단·보고(FP32라 낮은 확률).
- 데이터 로딩 병목 시 `data.workers_per_gpu` 상향(CPU 코어 수 기준), 그래도 부족하면 동시 실험 수를 1로 줄이고 8 GPU 사용.
- 디스크: run당 best+last만 보관, 나머지 epoch ckpt는 평가 후 삭제(스크립트).

## 3. 사전 확인 필요(접속 후 D0에 확정)
CPU 코어/RAM(데이터 로더 수), 디스크 여유(데이터 475GB + runs 20GB), nuScenes full 존재 여부와 infos 포맷(bevfusion 포맷 필요), nvcc/driver 버전(conda cuda-toolkit 필요 여부), 컨테이너 권한(pip/conda 설치 가능 여부), 네트워크(pip/conda 채널 접근).

## v2 (2026-09-19) — 2단계 구조로 변경 (사용자 제공 실행계획 문서 반영)

- **Phase 1 스크리닝**: 16개 전부, **6 epoch 확정(2026-09-19 사용자 위임 → 오케스트레이터 제안 채택; LR 사이클도 6 epoch에 맞춤, 문서 원안 2 epoch 기각)**, 2×B200/실험, 4개 동시 → 4 wave. wave 구성은 문서 §9(B0/A1/A2/A3 → A4/C1/C2/C3 → C4/C5/C6/P1 → P2/P3/P4/FINAL).
- **Phase 2 최종**: Top-4(NDS 기준, mAP·안정성·s/iter·VRAM 보조) + B0·FINAL 강제 포함 → **20 epoch 표준 스케줄**, 4개 동시(2 GPU씩) 또는 2개 동시(4 GPU씩). 문서 원안의 6 epoch은 표준 수치가 아니어서 채택하지 않음.
- 실험 ID 별칭: B0, A1(dsvt), A2(wf), A3(dal), A4(gf), C1~C6, P1~P4, FINAL ↔ 4비트 이름. 런처·집계: `tools/ablation/launch_waves.py`, `aggregate.py` (SPEC_launcher.md).
- 시간은 step3 속도 실측(4 GPU c4_full s/iter) 후 확정. 문서의 14~23h는 미실측 추정.
