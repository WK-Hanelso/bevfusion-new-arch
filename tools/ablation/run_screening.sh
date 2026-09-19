#!/usr/bin/env bash
# 스크리닝 원샷 실행기 — 게이트(선택 ID 전부) → 런처 → 자동 집계. 화면에 60초마다 상태를 찍는다.
# 사용: bash tools/ablation/run_screening.sh <nuscenes_root> <ID ...>
#   env: GPUS_PER_JOB(기본 1)  PARALLEL(기본 8)  EPOCHS(기본 6)  LOAD_FROM_DSVT(기본 pretrained/dsvt_nuscenes_official_lidar.pth)
#        GPUS(기본 0,1,2,3,4,5,6,7; 게이트·런처 공통)  MASTER_PORT(기본 29500)  TAG(로그 접미사, 기본 없음)  RESUME=1(완료 run 건너뛰고 미완료 run은 최신 epoch ckpt에서 이어받기)
#   Ctrl-C 하면 게이트/런처/학습 프로세스를 전부 종료한다.
set -uo pipefail
ROOT="${1:?nuscenes_root}"; shift; IDS=("$@"); [ ${#IDS[@]} -eq 0 ] && { echo "IDs required"; exit 2; }
cd "$(dirname "$0")/../.."
GPJ="${GPUS_PER_JOB:-1}"; PAR="${PARALLEL:-8}"; EP="${EPOCHS:-6}"; DSVT="${LOAD_FROM_DSVT:-pretrained/dsvt_nuscenes_official_lidar.pth}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"; MPORT="${MASTER_PORT:-29500}"; TAG="${TAG:-}"; RESUME_FLAG=""; [ "${RESUME:-0}" = "1" ] && RESUME_FLAG="--resume"
RUNLOG="experiments/screening_run${TAG:+_$TAG}.log"; mkdir -p experiments
: > "$RUNLOG"
T0=$(date +%s)
elapsed() { local s=$(( $(date +%s) - T0 )); printf '%02d:%02d:%02d' $((s/3600)) $((s%3600/60)) $((s%60)); }

cleanup() {
  echo; echo "== 종료 요청: 게이트/런처/학습 프로세스 정리"
  [ -n "${PIPE_PID:-}" ] && kill -TERM -- -"$PIPE_PID" 2>/dev/null
  pkill -TERM -f "tools/ablation/launch_waves.py" 2>/dev/null
  pkill -TERM -f "tools/train_torchrun.py" 2>/dev/null
  sleep 3; pkill -KILL -f "tools/train_torchrun.py" 2>/dev/null
  exit 130
}
trap cleanup INT TERM

echo "== run_screening start $(date)  ids=${IDS[*]}  gpus=$GPUS gpus_per_job=$GPJ parallel=$PAR epochs=$EP port=$MPORT"
echo "== 상세 로그: $RUNLOG   run별 로그: experiments/runs/screening/<ID>/train.log"
setsid bash -c "
  GATE_GPUS='$GPUS' GATE_PER_JOB=$GPJ bash tools/ablation/gate_e2e.sh '$ROOT' ${IDS[*]} && \
  python tools/ablation/launch_waves.py --phase screening --ids ${IDS[*]} --gpus '$GPUS' --gpus-per-job $GPJ --parallel $PAR --epochs $EP \
    --master-port $MPORT $RESUME_FLAG --dataroot '$ROOT' --load-from-dsvt '$DSVT'
  echo \"PIPELINE_EXIT=\$?\"
" > "$RUNLOG" 2>&1 &
PIPE_PID=$!

LAST=0
while kill -0 "$PIPE_PID" 2>/dev/null; do
  sleep 5
  NOW=$(date +%s); [ $((NOW - LAST)) -lt 60 ] && continue; LAST=$NOW
  echo; echo "---- [$(elapsed) 경과] $(date +%H:%M:%S) ----"
  grep -E "^== |PASS |FAIL |^\[(GATE|START|COMPLETED|FAILED|STOP|SKIP|WARN)\]|PIPELINE_EXIT" "$RUNLOG" | tail -12 | cut -c1-140
  # 게이트가 끝나기 전에는(런처 [START] 없음) 게이트 진행을, 그 뒤에는 run 진행을 보여준다.
  if grep -q "^\[START\]" "$RUNLOG" 2>/dev/null && [ -d experiments/runs/screening ]; then
    for d in experiments/runs/screening/*/; do
      [ -d "$d" ] || continue
      id=$(basename "$d"); last=$(grep -hE 'Epoch \[' "$d/train.log" 2>/dev/null | tail -1 | sed 's/.*Epoch/Epoch/' | cut -c1-72)
      ev=$(grep -hE '^mAP: |^NDS: ' "$d/train.log" 2>/dev/null | tail -2 | tr '\n' ' ')
      err=$(grep -hE 'Error|error:' "$d/train.log" 2>/dev/null | grep -v UserWarning | tail -1 | cut -c1-70)
      printf '  %-6s %s %s %s\n' "$id" "$last" "$ev" "${err:+ERR: $err}"
    done
  elif [ -d experiments/gate ]; then
    for d in experiments/gate/run_*/; do
      [ -d "$d" ] || continue
      id=$(basename "$d" | sed 's/run_//'); last=$(tail -c 400 "$d/train.log" 2>/dev/null | tr '\r' '\n' | grep -E 'Epoch \[|/6019|mAP: |NDS: |Error|out of memory' | tail -1 | sed 's/.*Epoch/Epoch/' | cut -c1-80)
      printf '  gate %-6s %s\n' "$id" "$last"
    done
  fi
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | awk -F', ' '{printf "  gpu%s %s/%s ", $1, $2, $3} END {print ""}'
done

echo; echo "== 파이프라인 종료 $(date)  ($(elapsed) 경과)"
grep -E "PIPELINE_EXIT|^\[(COMPLETED|FAILED)\]|== gate" "$RUNLOG" | tail -20 | cut -c1-140
if grep -q "PIPELINE_EXIT=0" "$RUNLOG"; then
  echo "== 집계"; python tools/ablation/aggregate.py --phase screening --top-k 2 --force-include-b0-final 2>&1 | tail -30
fi
