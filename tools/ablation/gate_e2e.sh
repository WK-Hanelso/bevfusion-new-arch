#!/usr/bin/env bash
# E2E 게이트 — 본 실행 전에 "학습 → 체크포인트 → full val 평가(mAP/NDS)"를 실제로 완주한다.
# 통과 시 experiments/gate/PASS_<git rev>.txt 를 만들고, launch_waves.py는 이 마커가 없으면 실행을 거부한다.
# 사용: bash tools/ablation/gate_e2e.sh <nuscenes_root> [ID ...]   (기본 ID: B0 A1 A2 FINAL — 4개를 GPU 그룹별로 동시에)
#   env: GATE_GPUS(기본 0,1,2,3,4,5,6,7)  GATE_PER_JOB(그룹당 GPU 수, 기본 2; 1이면 8개 ID 동시)  GATE_TRAIN_SAMPLES(기본 512)  CONDA_ENV(기본 bevfusion-b200)
set -uo pipefail
ROOT="${1:?nuscenes_root}"; shift; IDS=("$@"); [ ${#IDS[@]} -eq 0 ] && IDS=(B0 A1 A2 FINAL)
cd "$(dirname "$0")/../.."
ENV="${CONDA_ENV:-bevfusion-b200}"; GPUS="${GATE_GPUS:-0,1,2,3,4,5,6,7}"; N="${GATE_TRAIN_SAMPLES:-512}"; PER_JOB="${GATE_PER_JOB:-2}"
REV="$(git rev-parse HEAD)"; GATE=experiments/gate; MINI="$GATE/mini_nuscenes"
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
echo "== gate start $(date)  rev=${REV:0:8} ids=${IDS[*]} gpus=$GPUS train_samples=$N"

echo "== 1) mini dataset root (train subset $N, full val)"
mkdir -p "$MINI"
for d in samples sweeps maps v1.0-trainval; do [ -e "$MINI/$d" ] || ln -s "$ROOT/$d" "$MINI/$d"; done
[ -e "$MINI/nuscenes_infos_val.pkl" ] || ln -s "$ROOT/nuscenes_infos_val.pkl" "$MINI/nuscenes_infos_val.pkl"
[ -e "$ROOT/nuscenes_dbinfos_train.pkl" ] && [ ! -e "$MINI/nuscenes_dbinfos_train.pkl" ] && ln -s "$ROOT/nuscenes_dbinfos_train.pkl" "$MINI/nuscenes_dbinfos_train.pkl"
conda run -n "$ENV" --no-capture-output python - "$ROOT" "$MINI" "$N" <<'PY'
import pickle, sys
root, mini, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = pickle.load(open(f"{root}/nuscenes_infos_train.pkl", "rb"))
d["infos"] = d["infos"][:n]
pickle.dump(d, open(f"{mini}/nuscenes_infos_train.pkl", "wb"))
print("mini train infos:", len(d["infos"]))
PY

# GPU 목록을 PER_JOB개씩 잘라 그룹을 만들고, ID를 그룹에 병렬 배치한다(그룹 수만큼 동시 실행).
mapfile -t GPU_ARR < <(echo "$GPUS" | tr ',' '\n')
GROUPS_N=$(( ${#GPU_ARR[@]} / PER_JOB )); [ $GROUPS_N -lt 1 ] && GROUPS_N=1
run_one() {  # $1=ID $2=gpu list "a,b" $3=port
  local ID="$1" G="$2" PORT="$3"
  local CFG RUN USES_DSVT EXTRA=()
  CFG=$(conda run -n "$ENV" --no-capture-output python -c "
import sys; sys.path.insert(0,'tools/ablation')
from experiments import BY_ID; print(BY_ID['$ID'].config_path)")
  USES_DSVT=$(conda run -n "$ENV" --no-capture-output python -c "
import sys; sys.path.insert(0,'tools/ablation')
from experiments import BY_ID; print('yes' if BY_ID['$ID'].uses_dsvt else 'no')")
  [ "$USES_DSVT" = "yes" ] && EXTRA=(--load_from pretrained/dsvt_nuscenes_official_lidar.pth)
  # per-GPU batch <= 16 (legacy spconv limit); reach global batch 32 via gradient accumulation
  local SPG=$((32 / PER_JOB)) ACC=(); if [ "$SPG" -gt 16 ] && [ "${BEVFUSION_SPCONV:-legacy}" != "v2" ]; then ACC=(--optimizer_config.type GradientCumulativeOptimizerHook --optimizer_config.cumulative_iters $((SPG / 16))); SPG=16; fi
  RUN="$GATE/run_${ID}"; rm -rf "$RUN"; mkdir -p "$RUN"
  echo "== 2) [$ID] gpus=$G port=$PORT cfg=$CFG start $(date +%H:%M:%S)  (progress: tail -f $RUN/train.log)"
  CUDA_VISIBLE_DEVICES="$G" conda run -n "$ENV" --no-capture-output torchrun --master_port="$PORT" --nproc_per_node="$PER_JOB" \
    tools/train_torchrun.py "$CFG" --run-dir "$RUN" --max_epochs 1 --dataset_root "$MINI/" --seed 0 --fp16 None \
    --find_unused_parameters True --checkpoint_config.out_dir "$RUN/checkpoints" \
    --data.samples_per_gpu "$SPG" --data.workers_per_gpu 8 "${ACC[@]}" "${EXTRA[@]}" > "$RUN/train.log" 2>&1
  local RC=$? CKPT MAP NDS
  CKPT=$(find "$RUN/checkpoints" -name "epoch_1.pth" 2>/dev/null | head -1)
  MAP=$(grep -oE "mAP: [0-9.]+" "$RUN/train.log" | tail -1); NDS=$(grep -oE "NDS: [0-9.]+" "$RUN/train.log" | tail -1)
  if [ $RC -eq 0 ] && [ -n "$CKPT" ] && [ -n "$MAP" ] && [ -n "$NDS" ]; then
    echo "   PASS $ID $(date +%H:%M:%S) ckpt=$CKPT $MAP $NDS"; echo PASS > "$RUN/RESULT"
  else
    echo "   FAIL $ID $(date +%H:%M:%S) rc=$RC ckpt='${CKPT}' map='${MAP}' nds='${NDS}'"
    grep -hE "^\[rank0\]: [A-Za-z.]*(Error|Exception)|^[A-Za-z.]*(Error|Exception):|out of memory" "$RUN/train.log" | grep -v UserWarning | head -3 | cut -c1-200 | sed 's/^/      /'
    echo FAIL > "$RUN/RESULT"
  fi
}
FAIL=0; i=0
while [ $i -lt ${#IDS[@]} ]; do
  PIDS=()
  for g in $(seq 0 $((GROUPS_N - 1))); do
    [ $i -ge ${#IDS[@]} ] && break
    GL=$(IFS=,; echo "${GPU_ARR[*]:$((g * PER_JOB)):$PER_JOB}")
    run_one "${IDS[$i]}" "$GL" $((29650 + g)) & PIDS+=($!); i=$((i + 1))
  done
  wait "${PIDS[@]}"
done
for ID in "${IDS[@]}"; do [ "$(cat "$GATE/run_${ID}/RESULT" 2>/dev/null)" = "PASS" ] || FAIL=1; done

if [ $FAIL -eq 0 ]; then
  {
    echo "rev=$REV"; echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; echo "ids=${IDS[*]}"
    conda run -n "$ENV" --no-capture-output python -c "import torch, mmcv; print('torch='+torch.__version__); print('mmcv='+mmcv.__version__)"
  } > "$GATE/PASS_${REV}.txt"
  echo "== gate PASS → $GATE/PASS_${REV}.txt $(date)"
else
  rm -f "$GATE/PASS_${REV}.txt"; echo "== gate FAIL $(date)"; exit 1
fi
