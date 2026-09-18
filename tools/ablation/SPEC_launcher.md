# SPEC — ablation wave 런처 (B200 8장, 2단계: 스크리닝 → Top-4 full)

참고 문서: 사용자 제공 "BEVFusion New-Arch Experiment Execution Plan"(§7~§17). 우리 상태: 16 config는 `configs/nuscenes/det/ablation/dsvt{0|1}_wf{0|1}_gf{0|1}_dal{0|1}.yaml`(생성기 `tools/ablation/gen_configs.py`), 학습 진입점 `tools/train_torchrun.py`(torchrun), 원장 `experiments/LEDGER.md`.

## 요구
1. **실험 ID 별칭**: B0(0000), A1 dsvt(1000), A2 wf(0100), A3 dal(0001), A4 gf(0010), C1 dsvt+wf, C2 dsvt+dal, C3 dsvt+gf, C4 wf+dal, C5 wf+gf, C6 dal+gf, P1 dsvt+wf+dal, P2 dsvt+wf+gf, P3 dsvt+dal+gf, P4 wf+dal+gf, FINAL(1111). `tools/ablation/experiments.py`에 단일 매핑 테이블(ID ↔ 4비트 ↔ config 경로)로 두고 gen_configs·런처·집계가 공유. 기존 별칭(b0_legacy, e1~e4, c2~c4)은 유지.
2. **런처 `tools/ablation/launch_waves.py`**: 인자 `--phase screening|final`, `--epochs N`, `--gpus-per-job 2|4`, `--parallel 4|2`, `--ids ...`(기본 스크리닝=16개 전부), `--dataroot`, `--load-from-dsvt PATH`(dsvt=1인 실험에만 자동 적용), `--seed`, `--dry-run`(명령·배정만 출력), `--resume`(완료된 ID 건너뜀). 동작: wave 단위로 GPU 그룹 배정(`CUDA_VISIBLE_DEVICES`), `torchrun --nproc_per_node=<gpus>` 실행, 각 실험 디렉토리 `experiments/runs/<phase>/<ID>/{config.yaml(resolved), command.txt, env.txt, train.log, metrics.json, checkpoints/}` 생성, wave 완료 대기 후 다음 wave. 실패 감지: 프로세스 비정상 종료, 로그의 NaN/Inf/`CUDA error`/OOM → `status=failed` 기록 후 해당 슬롯만 다음 실험으로. 재시작은 하지 않음(문서 §11).
3. **epoch 축소 스케줄**: `--epochs N`은 `max_epochs`와 LR/momentum cyclic 스케줄의 길이를 함께 N으로 맞춘다(torchpack config override로, config 파일 수정 없음). 기본 스크리닝 6, 최종 20.
4. **집계 `tools/ablation/aggregate.py`**: 각 run의 mmdet3d 평가 로그에서 epoch별 mAP/NDS/mATE/mASE/mAOE/mAVE/mAAE, s/iter, peak VRAM(로그의 memory), wall time, NaN 여부, ckpt 경로, git commit, seed를 파싱해 `results/screening_summary.csv`·`results/final_ablation.csv`·Markdown 표(B0 대비 ΔmAP/ΔNDS/Δs·iter/ΔVRAM 포함) 생성. `--top-k 4 --by NDS`로 Top-4 선정 출력(B0·FINAL 강제 포함 옵션).
5. **환경 캡처**: 런처 시작 시 `experiments/runs/<phase>/env_nvidia_smi.txt`, `env_nvcc.txt`, `env_pip_freeze.txt`(conda run -n bevfusion-b200).
6. 검증(로컬, GPU·mmcv 없음): `--dry-run`으로 16개 배정·명령 출력 테스트, 집계는 가짜 로그 fixture로 파싱 테스트, 기존 pytest 유지. `tools/ablation/README.md`에 사용법. EXEC: `tools/ablation/EXEC_launcher.md`. commit 금지.
