# SPEC — ablation wave 런처 (B200 8장, 2단계: 스크리닝 → Top-4 full)

참고 문서: 사용자 제공 "BEVFusion New-Arch Experiment Execution Plan"(§7~§17). 우리 상태: 16 config는 `configs/nuscenes/det/ablation/dsvt{0|1}_wf{0|1}_gf{0|1}_dal{0|1}.yaml`(생성기 `tools/ablation/gen_configs.py`), 학습 진입점 `tools/train_torchrun.py`(torchrun), 원장 `experiments/LEDGER.md`.

## 요구
1. **실험 ID 별칭**: B0(0000), A1 dsvt(1000), A2 wf(0100), A3 dal(0001), A4 gf(0010), C1 dsvt+wf, C2 dsvt+dal, C3 dsvt+gf, C4 wf+dal, C5 wf+gf, C6 dal+gf, P1 dsvt+wf+dal, P2 dsvt+wf+gf, P3 dsvt+dal+gf, P4 wf+dal+gf, FINAL(1111). `tools/ablation/experiments.py`에 단일 매핑 테이블(ID ↔ 4비트 ↔ config 경로)로 두고 gen_configs·런처·집계가 공유. 기존 별칭(b0_legacy, e1~e4, c2~c4)은 유지.
2. **런처 `tools/ablation/launch_waves.py`**: 인자 `--phase screening|final`, `--epochs N`, `--gpus-per-job 2|4`, `--parallel 4|2`, `--ids ...`(기본 스크리닝=16개 전부), `--dataroot`, `--load-from-dsvt PATH`(dsvt=1인 실험에만 자동 적용), `--seed`, `--dry-run`(명령·배정만 출력), `--resume`(완료된 ID 건너뜀). 동작: wave 단위로 GPU 그룹 배정(`CUDA_VISIBLE_DEVICES`), `torchrun --nproc_per_node=<gpus>` 실행, 각 실험 디렉토리 `experiments/runs/<phase>/<ID>/{config.yaml(resolved), command.txt, env.txt, train.log, metrics.json, checkpoints/}` 생성, wave 완료 대기 후 다음 wave. 실패 감지: 프로세스 비정상 종료, 로그의 NaN/Inf/`CUDA error`/OOM → `status=failed` 기록 후 해당 슬롯만 다음 실험으로. 재시작은 하지 않음(문서 §11).
3. **epoch 축소 스케줄**: `--epochs N`은 `max_epochs`와 LR/momentum cyclic 스케줄의 길이를 함께 N으로 맞춘다(torchpack config override로, config 파일 수정 없음). 기본 스크리닝 6, 최종 20.
4. **집계 `tools/ablation/aggregate.py`**: 각 run의 mmdet3d 평가 로그에서 epoch별 mAP/NDS/mATE/mASE/mAOE/mAVE/mAAE, s/iter, peak VRAM(로그의 memory), wall time, NaN 여부, ckpt 경로, git commit, seed를 파싱해 `results/screening_summary.csv`·`results/final_ablation.csv`·Markdown 표(B0 대비 ΔmAP/ΔNDS/Δs·iter/ΔVRAM 포함) 생성. `--top-k 4 --by NDS`로 Top-4 선정 출력(B0·FINAL 강제 포함 옵션).
5. **환경 캡처**: 런처 시작 시 `experiments/runs/<phase>/env_nvidia_smi.txt`, `env_nvcc.txt`, `env_pip_freeze.txt`(conda run -n bevfusion-b200).
6. 검증(로컬, GPU·mmcv 없음): `--dry-run`으로 16개 배정·명령 출력 테스트, 집계는 가짜 로그 fixture로 파싱 테스트, 기존 pytest 유지. `tools/ablation/README.md`에 사용법. EXEC: `tools/ablation/EXEC_launcher.md`. commit 금지.

## 7. 풀 스케줄러 (2026-09-19 추가, wave 배리어 대체)

동기: B200 실측에서 wave 1의 3개 슬롯이 즉시 실패하고 1개(A2, 10h)만 살아남자 GPU 6장이 10시간 유휴. 사용자 규칙 "GPU가 쉬면 안 됨".

요구:
1. `launch_waves.py`는 **wave 배리어를 제거**하고 GPU 그룹 풀로 스케줄한다. 대기열 = 선택된 실험(기존 순서 유지). 그룹이 비는 즉시(완료·실패·fatal 패턴 kill) 대기열의 다음 실험을 그 그룹에 투입. 모든 실험이 끝나면 종료.
2. `--gpus 0,1,2,3,6,7` (기본 0~7): 사용할 GPU 목록. 목록을 `--gpus-per-job` 크기로 앞에서부터 잘라 그룹을 만든다(6개·2/job → 3그룹). `--parallel`은 그룹 수 상한으로만 동작(기본 = 그룹 수).
3. 출력: 투입 시 `[START] slot=<g> gpus=<list> id=<ID> (queued=<n>)`, 종료 시 기존 `[COMPLETED]/[FAILED] … wall=`을 **즉시** 출력(배리어 대기 없이). `[PLAN]`은 dry-run과 실행 시작 시 대기열 전체를 1회 출력.
4. metrics.json의 `wave` 필드는 유지하되 의미를 "투입 순번"으로 바꾼다(호환).
5. 각 슬롯의 `--master_port`는 그룹 인덱스 기준(29500+g)이며 재사용 시 이전 프로세스 종료를 확인한 뒤 투입(포트 충돌 방지, 2초 대기 후 재시도 최대 5회).
6. `--resume`: status=completed 건너뜀(기존). status=running인 run 디렉터리가 있고 해당 프로세스가 없으면(pid 파일 `launcher.pid` 기록·확인) 재실행 대상.
7. 테스트: 기존 80개 유지 + (a) 3그룹·5개 실험에서 첫 실험이 즉시 실패하면 4번째 실험이 같은 그룹에 곧바로 투입되는지(subprocess를 가짜 스크립트로 대체), (b) `--gpus 0,1,2,3,6,7` 그룹 분할, (c) dry-run 출력 형식.
8. README 갱신. EXEC_launcher.md에 §7 절 추가.
