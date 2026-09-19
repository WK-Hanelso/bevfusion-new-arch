# SPEC — 런처 v3: 명령 하나로 게이트 → 16개 스크리닝 → 집계 (정합성 재설계)

## 동기 (2026-09-19)
스크리닝 운영이 런처 2개, 고아 프로세스, 대기 체인, "pull 금지" 주의사항으로 얽힘. 사용자: "반창고 덕지덕지, 정합하게 하라. 전부 종료하고 계획 제대로 짜서 돌려라."
목표: **런처 하나가 모든 것을 소유**하고, 사용자는 명령 한 줄만 실행한다. 예외·수동 체인·주의사항 0개.

## 요구
1. **단일 진입점** `python tools/ablation/launch_waves.py --phase screening --dataroot <root> --load-from-dsvt <ckpt>`
   가 순서대로 수행: (a) 게이트(필요 시) → (b) 16개 풀 스케줄 → (c) 완료 시 `aggregate.py --phase screening --top-k 4 --force-include-b0-final` 자동 실행, `TOP_K_IDS=` 출력.
   `--phase final --ids …`도 동일 구조(게이트 → 실행 → 집계).
2. **게이트 키 = 모델 코드 해시**(커밋 아님): `mmdet3d/**/*.py`, `configs/**/*.yaml`, `tools/train.py`, `tools/train_torchrun.py`, `tools/data_converter/**`, `docker/requirements-cu128.txt`, `docker/b200_conda_env.yml` 내용의 sha256(정렬된 상대경로+내용).
   마커 `experiments/gate/PASS_<hash12>.txt`. 런처(tools/ablation/*)만 바뀌면 재게이트 불필요. 마커가 없으면 런처가 **게이트를 직접 실행**(기존 `gate_e2e.sh` 로직을 파이썬으로 흡수: 512샘플 1 epoch → ckpt → full val mAP/NDS, B0/A1/A2/FINAL 4개를 2-GPU 그룹 4개에 병렬). 게이트 실패 시 즉시 종료(코드 3), 어떤 실험도 시작하지 않음. `--skip-gate`는 유지하되 사용 시 `[GATE] SKIPPED` 경고.
3. **프로세스 소유**: 모든 자식은 `start_new_session=True`로 시작, 런처 종료(Ctrl-C/SIGTERM) 시 전 슬롯 프로세스 그룹 SIGTERM→5s→SIGKILL. 런처가 죽으면 학습도 죽는다(고아 금지). 시작 시 `experiments/runs/<phase>/LAUNCHER.pid` 기록, 이미 살아 있는 런처가 있으면 거부.
4. **실패 처리**: 종료 코드 ≠0 이고 wall < 120s(빠른 실패: 포트·환경)이면 30s 후 **같은 슬롯에서 1회 자동 재시도**(`attempt=2`). 재시도도 실패하면 `[FAILED]`로 기록하고 큐를 계속 소진. NaN/Inf 감지는 학습 loss 필드만(기존). 포트: 슬롯별 고정 + 시작 전 bind 확인(기존).
5. **가시성**: 포그라운드 화면에 (a) 시작/종료 이벤트(기존 태그), (b) **5분마다 heartbeat 한 줄/슬롯**: `[HB] slot=k id=… epoch=e/E iter=i/N time=… mem=… loss=…`(train.log 마지막 Epoch 줄에서 파싱), (c) 큐 잔여 수. `tools/ablation/status.py`: 같은 정보를 1회 출력(다른 창용).
6. **상태 파일** `experiments/runs/<phase>/STATE.json`: 각 ID의 status(queued/running/completed/failed), attempts, slot, gpus, start/end, last_epoch, last_metrics. 런처가 갱신, aggregate가 참조 가능. `--resume`: completed 건너뜀, failed/미완료는 최신 `epoch_*.pth`에서 `--resume_from`(기존).
7. **기본값(B200)**: screening = 2 GPU/job, jobs-per-gpu 2, parallel 8, 6 epoch, samples_per_gpu 16(전역 32). final = 4 GPU/job, jobs-per-gpu 1, parallel 2, 20 epoch, samples_per_gpu 8. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
8. **정리**: `gate_e2e.sh`는 파이썬 게이트 호출 래퍼로 축소(또는 삭제하고 README 갱신). README에 "운영 명령은 하나" 절. 기존 테스트 유지 + 신규: 해시 키 안정성(런처 파일 변경 시 불변, mmdet3d 파일 변경 시 변함), 빠른 실패 재시도 1회, heartbeat 파싱, 런처 중복 실행 거부, 종료 시 그룹 kill(가짜 subprocess).
9. 산출물: launch_waves.py, gate.py(신규, 게이트 로직), status.py, tests, README, `EXEC_launcher_v3.md`.

## 10. 추가 결정 (2026-09-19, 사용자 확정)
- **스크리닝 범위 = 핵심 8개**: `B0 A1 A2 A3 A4 C1 P2 FINAL`, 6 epoch, 레시피 동일(학습률 변경 없음). `--preset core8` 로 선택 가능(= 위 8개 ID).
- **GPU 1장당 실험 1개**: `--gpus-per-job` 허용값 `1,2,4`. B200 스크리닝 기본 = `gpus_per_job 1, jobs_per_gpu 1, parallel 8, samples_per_gpu 32`(전역 배치 32 유지, `GLOBAL_BATCH // gpus_per_job`). `--nproc_per_node=1`일 때도 torchrun 경로 그대로 사용.
- **게이트는 선택된 ID 전부**를 대상으로(4개 고정 아님), 본 실행과 같은 `gpus_per_job`·`samples_per_gpu`로 GPU 그룹에 병렬. 즉 core8이면 8개 게이트가 8장에서 동시에 돈다(약 15분). 게이트 결과(각 ID의 mAP/NDS, 최대 메모리)는 STATE.json에 기록. 게이트에서 OOM이 나면 그 ID는 `[GATE-FAIL id=… reason=OOM]`로 표시하고 전체 실행을 시작하지 않는다(사용자 판단 필요 → 종료 코드 3).
- **최종 단계 기본**(내일): `--phase final --ids <Top-2>`: gpus_per_job 4, parallel 2, 20 epoch, samples_per_gpu 8.
- 마커 파일명은 모델 코드 해시 + 선택 ID 집합 해시를 함께 포함: `PASS_<model12>_<ids8>.txt`.
