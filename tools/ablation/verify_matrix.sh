#!/usr/bin/env bash
# Run the pre-flight verification matrix (M1--M7; M8/spconv2 is intentionally excluded).
# Usage: bash tools/ablation/verify_matrix.sh <nuscenes_root> [--dry-run]
set -uo pipefail

usage() {
  cat <<'EOF'
Usage: bash tools/ablation/verify_matrix.sh <nuscenes_root> [--dry-run]

Environment:
  VERIFY_GPUS          eight comma-separated GPU IDs (default: 0,1,2,3,4,5,6,7)
  CONDA_ENV            conda environment (default: bevfusion-b200)
  LOAD_FROM_DSVT       DSVT initialization checkpoint
  VERIFY_TRAIN_SAMPLES mini-train subset size (default: 512)
  VERIFY_STATUS_SECS   status interval in seconds (default: 60)
EOF
}

DRY_RUN=0
NUSCENES_ROOT=""
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --with-spconv2)
      echo "error: M8/spconv2 is excluded from this implementation" >&2
      exit 2
      ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "error: unknown option: $arg" >&2; usage >&2; exit 2 ;;
    *)
      if [ -n "$NUSCENES_ROOT" ]; then
        echo "error: unexpected argument: $arg" >&2
        usage >&2
        exit 2
      fi
      NUSCENES_ROOT="$arg"
      ;;
  esac
done
if [ -z "$NUSCENES_ROOT" ]; then
  usage >&2
  exit 2
fi

cd "$(dirname "$0")/../.."
CONDA_ENV="${CONDA_ENV:-bevfusion-b200}"
VERIFY_GPUS="${VERIFY_GPUS:-0,1,2,3,4,5,6,7}"
TRAIN_SAMPLES="${VERIFY_TRAIN_SAMPLES:-512}"
STATUS_SECS="${VERIFY_STATUS_SECS:-60}"
DSVT_CKPT="${LOAD_FROM_DSVT:-pretrained/dsvt_nuscenes_official_lidar.pth}"
REV="$(git rev-parse HEAD 2>/dev/null || printf unknown)"
REV_SHORT="${REV:0:8}"
RUN_TAG="run_${REV_SHORT}_$(date -u +%Y%m%dT%H%M%SZ)_$$"
VERIFY_ROOT="experiments/verify"
RUN_ROOT="$VERIFY_ROOT/$RUN_TAG"
MINI_ROOT="$RUN_ROOT/mini_nuscenes"
M1_RUNS_ROOT="$RUN_ROOT/m1_runs"
M1_PHASE_DIR="$M1_RUNS_ROOT/screening"
RESULTS_DIR="$RUN_ROOT/results"
MARKER="$VERIFY_ROOT/PASS_${REV}.txt"
MASTER_PORT_BASE="${VERIFY_MASTER_PORT_BASE:-31600}"

if ! [[ "$TRAIN_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: VERIFY_TRAIN_SAMPLES must be a positive integer" >&2
  exit 2
fi
if ! [[ "$STATUS_SECS" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: VERIFY_STATUS_SECS must be a positive integer" >&2
  exit 2
fi
IFS=',' read -r -a GPU_IDS <<< "$VERIFY_GPUS"
if [ "${#GPU_IDS[@]}" -ne 8 ]; then
  echo "error: VERIFY_GPUS must contain exactly eight GPU IDs" >&2
  exit 2
fi
declare -A GPU_SEEN=()
for gpu in "${GPU_IDS[@]}"; do
  if ! [[ "$gpu" =~ ^[0-9]+$ ]] || [ -n "${GPU_SEEN[$gpu]:-}" ]; then
    echo "error: VERIFY_GPUS must contain eight distinct non-negative integers" >&2
    exit 2
  fi
  GPU_SEEN[$gpu]=1
done

declare -a IDS=()
declare -A CFG=()
declare -A USES_DSVT=()
while IFS='|' read -r experiment_id config_path uses_dsvt; do
  IDS+=("$experiment_id")
  CFG["$experiment_id"]="$config_path"
  USES_DSVT["$experiment_id"]="$uses_dsvt"
done < <(
  PYTHONPATH=tools/ablation python - <<'PY'
from experiments import EXPERIMENTS
for item in EXPERIMENTS:
    print(f"{item.experiment_id}|{item.config_path}|{int(item.uses_dsvt)}")
PY
)
if [ "${#IDS[@]}" -ne 16 ]; then
  echo "error: experiments.py did not provide exactly 16 experiments" >&2
  exit 2
fi

declare -a M2_IDS=(B0 A1 A2 A3 A4 C5 P2 FINAL)
declare -a PAIR_IDS=(B0 FINAL)
declare -a ACTIVE_PIDS=()

quote_command() {
  printf '%q ' "$@"
  printf '\n'
}

make_train_command() {
  local -n destination=$1
  local id="$2" run_dir="$3" nproc="$4" samples_per_gpu="$5"
  local cumulative_iters="$6" epochs="$7" port="$8" resume_from="$9"
  destination=(
    conda run -n "$CONDA_ENV" --no-capture-output
    torchrun "--master_port=$port" "--nproc_per_node=$nproc"
    tools/train_torchrun.py "${CFG[$id]}"
    --run-dir "$run_dir"
    --max_epochs "$epochs"
    --dataset_root "$MINI_ROOT/"
    --seed 0
    --fp16 None
    --find_unused_parameters True
    --checkpoint_config.out_dir "$run_dir/checkpoints"
    --checkpoint_config.max_keep_ckpts 2
    --data.samples_per_gpu "$samples_per_gpu"
    --data.workers_per_gpu 8
  )
  if [ "$cumulative_iters" -gt 1 ]; then
    destination+=(
      --optimizer_config.type GradientCumulativeOptimizerHook
      --optimizer_config.cumulative_iters "$cumulative_iters"
    )
  fi
  if [ -n "$resume_from" ]; then
    destination+=(--resume_from "$resume_from")
  elif [ "${USES_DSVT[$id]}" = "1" ]; then
    destination+=(--load_from "$DSVT_CKPT")
  fi
}

find_checkpoint() {
  local run_dir="$1" epoch="$2"
  find "$run_dir/checkpoints" -type f -name "epoch_${epoch}.pth" -print -quit 2>/dev/null
}

has_nonfinite_loss() {
  local log_file="$1"
  grep -Eiq '(loss([^:[:space:]]*)?|grad_norm)["'"'']?[[:space:]]*:[[:space:]]*[-+]?(nan|inf)\b|Loss is nan' "$log_file"
}

has_training_loss() {
  local log_file="$1"
  grep -Eiq 'loss([^:[:space:]]*)?["'"'']?[[:space:]]*:[[:space:]]*[-+]?[0-9.]' "$log_file"
}

write_result() {
  local result_file="$1" status="$2" detail="$3"
  printf '%s\t%s\n' "$status" "$detail" > "$result_file"
}

run_train_job() {
  local matrix="$1" id="$2" devices="$3" nproc="$4" samples_per_gpu="$5"
  local cumulative_iters="$6" epochs="$7" port="$8" run_dir="$9" resume_from="${10:-}"
  local log_file="$run_dir/train.log" result_file="$run_dir/RESULT"
  local rc=0 checkpoint="" map_count=0 detail=""
  local -a command=()
  mkdir -p "$run_dir/checkpoints"
  make_train_command command "$id" "$run_dir" "$nproc" "$samples_per_gpu" \
    "$cumulative_iters" "$epochs" "$port" "$resume_from"
  {
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$devices"
    quote_command "${command[@]}"
  } > "$run_dir/command.txt"
  echo "[START] $matrix/$id gpu=$devices port=$port"

  setsid env CUDA_VISIBLE_DEVICES="$devices" "${command[@]}" > "$log_file" 2>&1 &
  local launcher_pid=$!
  trap 'kill -TERM -- -"$launcher_pid" 2>/dev/null || true' TERM INT
  wait "$launcher_pid" || rc=$?
  trap - TERM INT

  map_count=$(grep -cE '(^|[^[:alnum:]_])mAP:[[:space:]]*[-+]?[0-9.]+' "$log_file" 2>/dev/null || true)
  checkpoint=$(find_checkpoint "$run_dir" "$epochs")
  case "$matrix" in
    M1)
      if [ "$rc" -eq 0 ] && [ -n "$checkpoint" ] && [ "$map_count" -ge 2 ] \
        && has_training_loss "$log_file" && ! has_nonfinite_loss "$log_file"; then
        write_result "$result_file" PASS "epoch_2.pth; mAP_count=$map_count; finite_loss"
      else
        detail="rc=$rc ckpt=$([ -n "$checkpoint" ] && echo yes || echo no) mAP_count=$map_count"
        has_training_loss "$log_file" || detail="$detail no_training_loss"
        has_nonfinite_loss "$log_file" && detail="$detail nonfinite_loss"
        write_result "$result_file" FAIL "$detail"
      fi
      ;;
    M2)
      if [ "$rc" -eq 0 ] && [ -n "$checkpoint" ] && \
        grep -Eiq 'resum(ed|ing|e).*(epoch[[:space:]_:=-]*1|epoch_1\.pth)' "$log_file" \
        && ! has_nonfinite_loss "$log_file"; then
        write_result "$result_file" PASS "resumed epoch 1; epoch_2.pth"
      else
        detail="rc=$rc ckpt=$([ -n "$checkpoint" ] && echo yes || echo no)"
        grep -Eiq 'resum(ed|ing|e).*(epoch[[:space:]_:=-]*1|epoch_1\.pth)' "$log_file" || detail="$detail no_resume_evidence"
        has_nonfinite_loss "$log_file" && detail="$detail nonfinite_loss"
        write_result "$result_file" FAIL "$detail"
      fi
      ;;
    M3|M4)
      if [ "$rc" -eq 0 ] && [ -n "$checkpoint" ] && [ "$map_count" -ge 1 ] \
        && has_training_loss "$log_file" && ! has_nonfinite_loss "$log_file"; then
        write_result "$result_file" PASS "epoch_1.pth; mAP_count=$map_count; finite_loss"
      else
        detail="rc=$rc ckpt=$([ -n "$checkpoint" ] && echo yes || echo no) mAP_count=$map_count"
        has_training_loss "$log_file" || detail="$detail no_training_loss"
        has_nonfinite_loss "$log_file" && detail="$detail nonfinite_loss"
        write_result "$result_file" FAIL "$detail"
      fi
      ;;
  esac
  echo "[$(cut -f1 "$result_file")] $matrix/$id $(cut -f2- "$result_file")"
}

print_status() {
  local label="$1" pid run_dir id last_line state
  echo "---- [$label status] $(date '+%F %T') ----"
  for pid in "${ACTIVE_PIDS[@]}"; do
    run_dir="${PID_RUN_DIR[$pid]}"
    id="${PID_LABEL[$pid]}"
    if kill -0 "$pid" 2>/dev/null; then state=RUNNING; else state=DONE; fi
    last_line=$(grep -hE 'Epoch \[|mAP: |NDS: |Error|Exception|out of memory' "$run_dir/train.log" 2>/dev/null | tail -1 | cut -c1-100)
    printf '  %-14s %-7s %s\n' "$id" "$state" "$last_line"
  done
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null \
    | awk -F', ' '{printf "  gpu%s %s/%s ", $1, $2, $3} END {if (NR) print ""}' || true
}

declare -A PID_RUN_DIR=()
declare -A PID_LABEL=()
wait_for_batch() {
  local label="$1" pid any_running elapsed=0
  while :; do
    any_running=0
    for pid in "${ACTIVE_PIDS[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then any_running=1; break; fi
    done
    [ "$any_running" -eq 0 ] && break
    sleep 1
    elapsed=$((elapsed + 1))
    if [ "$elapsed" -ge "$STATUS_SECS" ]; then
      print_status "$label"
      elapsed=0
    fi
  done
  for pid in "${ACTIVE_PIDS[@]}"; do wait "$pid" || true; done
  ACTIVE_PIDS=()
  PID_RUN_DIR=()
  PID_LABEL=()
}

launch_train() {
  local matrix="$1" id="$2" devices="$3" nproc="$4" spg="$5" cumulative="$6"
  local epochs="$7" port="$8" run_dir="$9" resume="${10:-}"
  run_train_job "$matrix" "$id" "$devices" "$nproc" "$spg" "$cumulative" \
    "$epochs" "$port" "$run_dir" "$resume" &
  local pid=$!
  ACTIVE_PIDS+=("$pid")
  PID_RUN_DIR["$pid"]="$run_dir"
  PID_LABEL["$pid"]="$matrix/$id"
}

run_m5_job() {
  local log_file="$1" exit_file="$2"
  local rc=0
  shift 2
  setsid env CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" BEVFUSION_ENTRY=test.py \
    "$@" > "$log_file" 2>&1 &
  local launcher_pid=$!
  trap 'kill -TERM -- -"$launcher_pid" 2>/dev/null || true' TERM INT
  wait "$launcher_pid" || rc=$?
  trap - TERM INT
  printf '%s\n' "$rc" > "$exit_file"
}

cleanup() {
  local pid
  echo "== interruption: terminating active verification jobs" >&2
  for pid in "${ACTIVE_PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "${ACTIVE_PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
  exit 130
}
trap cleanup INT TERM

aggregate_supports_verify() {
  PYTHONPATH=tools/ablation python - <<'PY' >/dev/null 2>&1
import aggregate
parser = aggregate.make_parser()
phase = next(action for action in parser._actions if action.dest == "phase")
raise SystemExit(0 if phase.choices and "verify" in phase.choices else 1)
PY
}

print_dry_run() {
  local round index slot id run_dir resume devices port
  local -a command=()
  echo "== DRY RUN: M1--M7 command plan (no files or processes are created)"
  echo "[SETUP] mini_root=$MINI_ROOT train_samples=$TRAIN_SAMPLES full_val=$NUSCENES_ROOT/nuscenes_infos_val.pkl"
  for round in 0 1; do
    echo "[BATCH] M1 round=$((round + 1)) (8 x 1 GPU)"
    for slot in "${!GPU_IDS[@]}"; do
      index=$((round * 8 + slot)); id="${IDS[$index]}"
      run_dir="$M1_PHASE_DIR/$id"; port=$((MASTER_PORT_BASE + slot))
      make_train_command command "$id" "$run_dir" 1 16 2 2 "$port" ""
      printf '  [M1/%s gpu=%s] ' "$id" "${GPU_IDS[$slot]}"
      printf 'CUDA_VISIBLE_DEVICES=%q ' "${GPU_IDS[$slot]}"; quote_command "${command[@]}"
    done
  done
  echo "[BATCH] M2 sampled resume (8 x 1 GPU)"
  for slot in "${!GPU_IDS[@]}"; do
    id="${M2_IDS[$slot]}"; run_dir="$RUN_ROOT/m2/$id"
    resume="$M1_PHASE_DIR/$id/checkpoints/**/epoch_1.pth"
    make_train_command command "$id" "$run_dir" 1 16 2 2 \
      "$((MASTER_PORT_BASE + 20 + slot))" "$resume"
    printf '  [M2/%s gpu=%s] ' "$id" "${GPU_IDS[$slot]}"
    printf 'CUDA_VISIBLE_DEVICES=%q ' "${GPU_IDS[$slot]}"; quote_command "${command[@]}"
  done
  echo "[BATCH] M3 B0+FINAL (2 x 4 GPU)"
  for slot in 0 1; do
    id="${PAIR_IDS[$slot]}"
    devices=$(IFS=,; echo "${GPU_IDS[*]:$((slot * 4)):4}")
    run_dir="$RUN_ROOT/m3/$id"
    make_train_command command "$id" "$run_dir" 4 8 1 1 \
      "$((MASTER_PORT_BASE + 40 + slot))" ""
    printf '  [M3/%s gpu=%s] ' "$id" "$devices"
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$devices"; quote_command "${command[@]}"
  done
  echo "[BATCH] M4 B0+FINAL (2 x 2 GPU)"
  for slot in 0 1; do
    id="${PAIR_IDS[$slot]}"
    devices=$(IFS=,; echo "${GPU_IDS[*]:$((slot * 2)):2}")
    run_dir="$RUN_ROOT/m4/$id"
    make_train_command command "$id" "$run_dir" 2 16 1 1 \
      "$((MASTER_PORT_BASE + 50 + slot))" ""
    printf '  [M4/%s gpu=%s] ' "$id" "$devices"
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$devices"; quote_command "${command[@]}"
  done
  echo "[M5] evaluation-only through BEVFUSION_ENTRY=test.py (M1 B0 epoch_1 fallback)"
  quote_command env "CUDA_VISIBLE_DEVICES=${GPU_IDS[0]}" BEVFUSION_ENTRY=test.py \
    conda run -n "$CONDA_ENV" --no-capture-output torchrun \
    "--master_port=$((MASTER_PORT_BASE + 60))" --nproc_per_node=1 tools/train_torchrun.py \
    "${CFG[B0]}" "$M1_PHASE_DIR/B0/checkpoints/**/epoch_1.pth" --eval bbox \
    --cfg-options "data.test.dataset_root=$MINI_ROOT/" \
    "data.test.ann_file=$MINI_ROOT/nuscenes_infos_val.pkl" data.workers_per_gpu=8
  echo "[M6] PYTHONPATH=. python -m pytest -q tools/ablation/tests"
  if aggregate_supports_verify; then
    echo "[M7] python tools/ablation/aggregate.py --phase verify --runs-root $M1_RUNS_ROOT --output-dir $RESULTS_DIR --top-k 16"
  else
    echo "[M7 fallback] python tools/ablation/aggregate.py --phase screening --runs-root $M1_RUNS_ROOT --output-dir $RESULTS_DIR --top-k 16"
  fi
  echo "[PASS] created only if every M1--M7 acceptance check passes: $MARKER"
}

if [ "$DRY_RUN" -eq 1 ]; then
  print_dry_run
  exit 0
fi

if [ ! -d "$NUSCENES_ROOT" ]; then
  echo "error: nuScenes root is not a directory: $NUSCENES_ROOT" >&2
  exit 2
fi
NUSCENES_ROOT=$(cd "$NUSCENES_ROOT" && pwd)
for required in samples sweeps maps v1.0-trainval nuscenes_infos_train.pkl nuscenes_infos_val.pkl; do
  if [ ! -e "$NUSCENES_ROOT/$required" ]; then
    echo "error: missing nuScenes input: $NUSCENES_ROOT/$required" >&2
    exit 2
  fi
done
if [ ! -f "$DSVT_CKPT" ]; then
  echo "error: DSVT checkpoint not found: $DSVT_CKPT" >&2
  exit 2
fi

mkdir -p "$MINI_ROOT" "$M1_PHASE_DIR" "$RESULTS_DIR"
if [ -f "$MARKER" ]; then
  mv "$MARKER" "$RUN_ROOT/PREVIOUS_PASS_${REV}.txt"
fi
for item in samples sweeps maps v1.0-trainval; do
  ln -s "$NUSCENES_ROOT/$item" "$MINI_ROOT/$item"
done
ln -s "$NUSCENES_ROOT/nuscenes_infos_val.pkl" "$MINI_ROOT/nuscenes_infos_val.pkl"
if [ -e "$NUSCENES_ROOT/nuscenes_dbinfos_train.pkl" ]; then
  ln -s "$NUSCENES_ROOT/nuscenes_dbinfos_train.pkl" "$MINI_ROOT/nuscenes_dbinfos_train.pkl"
fi
conda run -n "$CONDA_ENV" --no-capture-output python - \
  "$NUSCENES_ROOT/nuscenes_infos_train.pkl" "$MINI_ROOT/nuscenes_infos_train.pkl" "$TRAIN_SAMPLES" <<'PY'
import pickle
import sys

source, destination, count = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(source, "rb") as handle:
    data = pickle.load(handle)
data["infos"] = data["infos"][:count]
with open(destination, "wb") as handle:
    pickle.dump(data, handle)
print(f"mini train infos: {len(data['infos'])}")
PY

echo "== verify start $(date) rev=$REV_SHORT run=$RUN_ROOT"
echo "== M1: all 16 configs, two rounds of eight 1-GPU jobs"
for round in 0 1; do
  for slot in "${!GPU_IDS[@]}"; do
    index=$((round * 8 + slot)); id="${IDS[$index]}"
    launch_train M1 "$id" "${GPU_IDS[$slot]}" 1 16 2 2 \
      "$((MASTER_PORT_BASE + slot))" "$M1_PHASE_DIR/$id"
  done
  wait_for_batch "M1 round $((round + 1))"
done

echo "== M2: resume eight representative configs from M1 epoch_1"
for slot in "${!GPU_IDS[@]}"; do
  id="${M2_IDS[$slot]}"
  resume=$(find_checkpoint "$M1_PHASE_DIR/$id" 1)
  if [ -z "$resume" ]; then
    mkdir -p "$RUN_ROOT/m2/$id"
    write_result "$RUN_ROOT/m2/$id/RESULT" FAIL "M1 epoch_1.pth missing"
    continue
  fi
  launch_train M2 "$id" "${GPU_IDS[$slot]}" 1 16 2 2 \
    "$((MASTER_PORT_BASE + 20 + slot))" "$RUN_ROOT/m2/$id" "$resume"
done
wait_for_batch M2

echo "== M3: B0 and FINAL, four GPUs each"
for slot in 0 1; do
  id="${PAIR_IDS[$slot]}"
  devices=$(IFS=,; echo "${GPU_IDS[*]:$((slot * 4)):4}")
  launch_train M3 "$id" "$devices" 4 8 1 1 \
    "$((MASTER_PORT_BASE + 40 + slot))" "$RUN_ROOT/m3/$id"
done
wait_for_batch M3

echo "== M4: B0 and FINAL, two GPUs each"
for slot in 0 1; do
  id="${PAIR_IDS[$slot]}"
  devices=$(IFS=,; echo "${GPU_IDS[*]:$((slot * 2)):2}")
  launch_train M4 "$id" "$devices" 2 16 1 1 \
    "$((MASTER_PORT_BASE + 50 + slot))" "$RUN_ROOT/m4/$id"
done
wait_for_batch M4

echo "== M5: evaluation-only test.py path"
M5_DIR="$RUN_ROOT/m5/B0"
mkdir -p "$M5_DIR"
M5_CKPT=$(find_checkpoint "$M1_PHASE_DIR/B0" 1)
if [ -z "$M5_CKPT" ]; then
  write_result "$M5_DIR/RESULT" FAIL "M1 B0 epoch_1.pth missing"
else
  M5_COMMAND=(
    conda run -n "$CONDA_ENV" --no-capture-output
    torchrun "--master_port=$((MASTER_PORT_BASE + 60))" --nproc_per_node=1
    tools/train_torchrun.py "${CFG[B0]}" "$M5_CKPT" --eval bbox
    --cfg-options "data.test.dataset_root=$MINI_ROOT/"
    "data.test.ann_file=$MINI_ROOT/nuscenes_infos_val.pkl"
    data.workers_per_gpu=8
  )
  {
    printf 'CUDA_VISIBLE_DEVICES=%q BEVFUSION_ENTRY=test.py ' "${GPU_IDS[0]}"
    quote_command "${M5_COMMAND[@]}"
  } > "$M5_DIR/command.txt"
  run_m5_job "$M5_DIR/train.log" "$M5_DIR/EXIT_CODE" "${M5_COMMAND[@]}" &
  m5_pid=$!
  ACTIVE_PIDS=("$m5_pid"); PID_RUN_DIR["$m5_pid"]="$M5_DIR"; PID_LABEL["$m5_pid"]="M5/B0"
  wait_for_batch M5
  m5_rc=$(cat "$M5_DIR/EXIT_CODE" 2>/dev/null || echo 127)
  if [ "$m5_rc" -eq 0 ] && grep -Eq '(^|[^[:alnum:]_])mAP:[[:space:]]*[-+]?[0-9.]+' "$M5_DIR/train.log"; then
    write_result "$M5_DIR/RESULT" PASS "test.py completed with mAP (M1 checkpoint fallback)"
  else
    write_result "$M5_DIR/RESULT" FAIL "rc=$m5_rc or mAP missing"
  fi
fi

echo "== M6: ablation unit tests"
M6_DIR="$RUN_ROOT/m6"; mkdir -p "$M6_DIR"
m6_rc=0
PYTHONPATH=. python -m pytest -q tools/ablation/tests > "$M6_DIR/pytest.log" 2>&1 || m6_rc=$?
if [ "$m6_rc" -eq 0 ]; then
  write_result "$M6_DIR/RESULT" PASS "pytest passed"
else
  write_result "$M6_DIR/RESULT" FAIL "pytest rc=$m6_rc"
fi

echo "== M7: aggregate M1 logs"
M7_DIR="$RUN_ROOT/m7"; mkdir -p "$M7_DIR"
ln -s screening "$M1_RUNS_ROOT/verify"
aggregate_phase=screening
aggregate_label="screening fallback"
if aggregate_supports_verify; then
  aggregate_phase=verify
  aggregate_label=verify
fi
m7_rc=0
python tools/ablation/aggregate.py --phase "$aggregate_phase" \
  --runs-root "$M1_RUNS_ROOT" --output-dir "$RESULTS_DIR" --top-k 16 \
  > "$M7_DIR/aggregate.log" 2>&1 || m7_rc=$?
csv_count=$(find "$RESULTS_DIR" -maxdepth 1 -type f -name '*.csv' | wc -l)
md_count=$(find "$RESULTS_DIR" -maxdepth 1 -type f -name '*.md' | wc -l)
if [ "$m7_rc" -eq 0 ] && [ "$csv_count" -ge 1 ] && [ "$md_count" -ge 1 ] \
  && grep -Eq '^TOP_K_IDS=.+$' "$M7_DIR/aggregate.log"; then
  write_result "$M7_DIR/RESULT" PASS "$aggregate_label; CSV/MD/TOP_K_IDS"
else
  write_result "$M7_DIR/RESULT" FAIL "$aggregate_label rc=$m7_rc csv=$csv_count md=$md_count"
fi

declare -A MATRIX_STATUS=()
declare -A MATRIX_DETAIL=()
summarize_group() {
  local matrix="$1" expected="$2" pattern="$3" result status detail pass_count=0 total=0
  while IFS= read -r result; do
    total=$((total + 1))
    IFS=$'\t' read -r status detail < "$result"
    [ "$status" = PASS ] && pass_count=$((pass_count + 1))
  done < <(find "$RUN_ROOT" -path "$pattern" -type f -name RESULT | sort)
  if [ "$total" -eq "$expected" ] && [ "$pass_count" -eq "$expected" ]; then
    MATRIX_STATUS["$matrix"]=PASS
  else
    MATRIX_STATUS["$matrix"]=FAIL
  fi
  MATRIX_DETAIL["$matrix"]="$pass_count/$expected passed (found $total results)"
}

summarize_group M1 16 "$M1_PHASE_DIR/*/RESULT"
summarize_group M2 8 "$RUN_ROOT/m2/*/RESULT"
summarize_group M3 2 "$RUN_ROOT/m3/*/RESULT"
summarize_group M4 2 "$RUN_ROOT/m4/*/RESULT"
for matrix in M5 M6 M7; do
  case "$matrix" in
    M5) result="$M5_DIR/RESULT" ;;
    M6) result="$M6_DIR/RESULT" ;;
    M7) result="$M7_DIR/RESULT" ;;
  esac
  if [ -f "$result" ]; then
    IFS=$'\t' read -r status detail < "$result"
    MATRIX_STATUS["$matrix"]="$status"; MATRIX_DETAIL["$matrix"]="$detail"
  else
    MATRIX_STATUS["$matrix"]=FAIL; MATRIX_DETAIL["$matrix"]="RESULT missing"
  fi
done

echo
echo '| Matrix | Status | Detail |'
echo '|---|---|---|'
all_pass=1
for matrix in M1 M2 M3 M4 M5 M6 M7; do
  printf '| %s | %s | %s |\n' "$matrix" "${MATRIX_STATUS[$matrix]}" "${MATRIX_DETAIL[$matrix]}"
  [ "${MATRIX_STATUS[$matrix]}" = PASS ] || all_pass=0
done

if [ "$all_pass" -eq 1 ]; then
  mkdir -p "$VERIFY_ROOT"
  {
    echo "status=PASS"
    echo "rev=$REV"
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "run_root=$RUN_ROOT"
    echo "matrices=M1 M2 M3 M4 M5 M6 M7"
    echo "m2_ids=${M2_IDS[*]}"
    echo "aggregate_phase=$aggregate_phase"
  } > "$MARKER"
  echo "== verification PASS: $MARKER"
else
  echo "== verification FAIL: no PASS marker written"
  exit 1
fi
