#!/usr/bin/env bash
# E2E 게이트 — 본 실행 전에 "학습 → 체크포인트 → full val 평가(mAP/NDS)"를 실제로 완주한다.
# 통과 시 experiments/gate/PASS_<git rev>.txt 를 만들고, launch_waves.py는 이 마커가 없으면 실행을 거부한다.
# 사용: bash tools/ablation/gate_e2e.sh <nuscenes_root> [ID ...]   (기본 ID: B0 FINAL)
#   env: GATE_GPUS(기본 0,1)  GATE_TRAIN_SAMPLES(기본 256)  CONDA_ENV(기본 bevfusion-b200)
set -uo pipefail
ROOT="${1:?nuscenes_root}"; shift; IDS=("$@"); [ ${#IDS[@]} -eq 0 ] && IDS=(B0 FINAL)
cd "$(dirname "$0")/../.."
ENV="${CONDA_ENV:-bevfusion-b200}"; GPUS="${GATE_GPUS:-0,1}"; N="${GATE_TRAIN_SAMPLES:-256}"
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

FAIL=0
for ID in "${IDS[@]}"; do
  CFG=$(conda run -n "$ENV" --no-capture-output python -c "
import sys; sys.path.insert(0,'tools/ablation')
from experiments import BY_ID; print(BY_ID['$ID'].config_path)")
  RUN="$GATE/run_${ID}"; rm -rf "$RUN"; mkdir -p "$RUN"
  echo "== 2) train 1 epoch + eval: $ID ($CFG) $(date)"
  EXTRA=()
  if conda run -n "$ENV" --no-capture-output python -c "
import sys; sys.path.insert(0,'tools/ablation')
from experiments import BY_ID; sys.exit(0 if BY_ID['$ID'].uses_dsvt else 1)"; then
    EXTRA=(--load_from pretrained/dsvt_nuscenes_official_lidar.pth); fi
  CUDA_VISIBLE_DEVICES="$GPUS" conda run -n "$ENV" --no-capture-output torchrun --master_port=29650 --nproc_per_node="$NPROC" \
    tools/train_torchrun.py "$CFG" --run-dir "$RUN" --max_epochs 1 --dataset_root "$MINI/" --seed 0 --fp16 None \
    --find_unused_parameters True --checkpoint_config.out_dir "$RUN/checkpoints" "${EXTRA[@]}" > "$RUN/train.log" 2>&1
  RC=$?
  CKPT=$(find "$RUN/checkpoints" -name "epoch_1.pth" 2>/dev/null | head -1)
  MAP=$(grep -oE "mAP: [0-9.]+" "$RUN/train.log" | tail -1); NDS=$(grep -oE "NDS: [0-9.]+" "$RUN/train.log" | tail -1)
  if [ $RC -eq 0 ] && [ -n "$CKPT" ] && [ -n "$MAP" ] && [ -n "$NDS" ]; then
    echo "   PASS $ID rc=$RC ckpt=$CKPT $MAP $NDS"
  else
    echo "   FAIL $ID rc=$RC ckpt='${CKPT}' map='${MAP}' nds='${NDS}'"; grep -E "Error|error" "$RUN/train.log" | tail -5; FAIL=1
  fi
done

if [ $FAIL -eq 0 ]; then
  {
    echo "rev=$REV"; echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; echo "ids=${IDS[*]}"
    conda run -n "$ENV" --no-capture-output python -c "import torch, mmcv; print('torch='+torch.__version__); print('mmcv='+mmcv.__version__)"
  } > "$GATE/PASS_${REV}.txt"
  echo "== gate PASS → $GATE/PASS_${REV}.txt $(date)"
else
  rm -f "$GATE/PASS_${REV}.txt"; echo "== gate FAIL $(date)"; exit 1
fi
