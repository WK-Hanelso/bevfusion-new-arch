#!/usr/bin/env bash
# 학습 프로브 — 긴 실행 전에 "배우는지"를 실데이터 서브셋(기본 6,000샘플 ≈ 1,900 iter)으로 확인한다.
# 판정: iter >= 1000 지점의 loss_heatmap < PROBE_PASS_HEATMAP(기본 2.0) 이면 PASS (B0 참조 1.74).
# 사용: bash tools/ablation/probe_learn.sh <nuscenes_root> "NAME|GPU|CONFIG|EXTRA ARGS" ["NAME2|GPU2|CONFIG2|EXTRA"...]
#   예: bash tools/ablation/probe_learn.sh ~/culee/data/nuscenes \
#        "S1_pre|7|configs/nuscenes/det/ablation/s1_dsvt_lidar.yaml|--load_from pretrained/dsvt_nuscenes_official_lidar.pth" \
#        "S1_scratch|1|configs/nuscenes/det/ablation/s1_dsvt_lidar.yaml|"
set -uo pipefail
ROOT="${1:?nuscenes_root}"; shift; SPECS=("$@"); [ ${#SPECS[@]} -eq 0 ] && { echo "variants required"; exit 2; }
cd "$(dirname "$0")/../.."
ENV=bevfusion-b200; N="${PROBE_TRAIN_SAMPLES:-6000}"; PASS_HM="${PROBE_PASS_HEATMAP:-2.0}"
PR=experiments/probe; MINI="$PR/mini_nuscenes"; mkdir -p "$MINI"
for d in samples sweeps maps v1.0-trainval; do [ -e "$MINI/$d" ] || ln -s "$ROOT/$d" "$MINI/$d"; done
[ -e "$MINI/nuscenes_infos_val.pkl" ] || ln -s "$ROOT/nuscenes_infos_val.pkl" "$MINI/nuscenes_infos_val.pkl"
# GT-Aug (ObjectPaste) assets, when the dataset has them
for f in nuscenes_dbinfos_train.pkl nuscenes_gt_database; do [ -e "$ROOT/$f" ] && [ ! -e "$MINI/$f" ] && ln -s "$ROOT/$f" "$MINI/$f"; done
conda run -n "$ENV" --no-capture-output python - "$ROOT" "$MINI" "$N" <<'PY'
import pickle, sys, os
root, mini, n = sys.argv[1], sys.argv[2], int(sys.argv[3]); out = f"{mini}/nuscenes_infos_train.pkl"
d = pickle.load(open(f"{root}/nuscenes_infos_train.pkl", "rb")); d["infos"] = d["infos"][:n]
pickle.dump(d, open(out, "wb")); print("probe train infos:", len(d["infos"]))
PY
NAMES=()
trap 'echo "== stop"; for n in "${NAMES[@]}"; do pkill -TERM -f "experiments/probe/$n/" 2>/dev/null; done; sleep 3; for n in "${NAMES[@]}"; do pkill -KILL -f "experiments/probe/$n/" 2>/dev/null; done; exit 130' INT TERM
PORT=29850
for spec in "${SPECS[@]}"; do
  IFS='|' read -r NAME GPU CFG EXTRA <<<"$spec"; NAMES+=("$NAME"); RUN="$PR/$NAME"; rm -rf "$RUN"; mkdir -p "$RUN"; PORT=$((PORT + 1))
  NPROC=$(echo "$GPU" | tr ',' '\n' | wc -l); SPG=$((32 / NPROC)); ACC=(); if [ "$SPG" -gt 16 ]; then ACC=(--optimizer_config.type GradientCumulativeOptimizerHook --optimizer_config.cumulative_iters $((SPG / 16))); SPG=16; fi
  echo "== [$NAME] gpus=$GPU nproc=$NPROC samples_per_gpu=$SPG cfg=$CFG extra='$EXTRA' port=$PORT"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$GPU" conda run -n "$ENV" --no-capture-output torchrun --master_port="$PORT" --nproc_per_node="$NPROC" \
    tools/train_torchrun.py "$CFG" --run-dir "$RUN" --max_epochs 1 --dataset_root "$MINI/" --seed 0 --fp16 None \
    --find_unused_parameters True --checkpoint_config.out_dir "$RUN/checkpoints" --checkpoint_config.interval 99 --evaluation.interval 99 \
    --data.samples_per_gpu "$SPG" --data.workers_per_gpu 8 "${ACC[@]}" \
    $EXTRA > "$RUN/train.log" 2>&1 &
done
T0=$(date +%s)
running() { for n in "${NAMES[@]}"; do pgrep -f "experiments/probe/$n/" >/dev/null && return 0; done; return 1; }
while running; do
  sleep 60; s=$(( $(date +%s) - T0 )); printf -- '---- [%02d:%02d 경과] ----\n' $((s/60)) $((s%60))
  for n in "${NAMES[@]}"; do
    line=$(grep -hE 'Epoch \[' "$PR/$n/train.log" 2>/dev/null | tail -1 | grep -oE "Epoch \[1\]\[[0-9]+/[0-9]+\]|loss_heatmap: [0-9.]+|loss_bbox: [0-9.]+|matched_ious: [0-9.]+" | tr '\n' ' ')
    err=$(grep -hE "Error" "$PR/$n/train.log" 2>/dev/null | grep -v Warning | tail -1 | cut -c1-80)
    printf '  %-14s %s %s\n' "$n" "$line" "${err:+ERR: $err}"
  done
done
echo "== probe done $(date)"
for n in "${NAMES[@]}"; do
  hm=$(grep -hE 'Epoch \[1\]\[1[0-9]{3}/' "$PR/$n/train.log" | head -1 | grep -oE "loss_heatmap: [0-9.]+" | grep -oE "[0-9.]+")
  # short runs (< 1000 iters): judge on the last logged value instead
  [ -z "$hm" ] && hm=$(grep -hE 'Epoch \[' "$PR/$n/train.log" | tail -1 | grep -oE "loss_heatmap: [0-9.]+" | grep -oE "[0-9.]+")
  last=$(grep -hE 'Epoch \[' "$PR/$n/train.log" | tail -1 | grep -oE "loss_heatmap: [0-9.]+|matched_ious: [0-9.]+" | tr '\n' ' ')
  verdict="FAIL"; [ -n "$hm" ] && awk -v a="$hm" -v b="$PASS_HM" 'BEGIN{exit !(a<b)}' && verdict="PASS"
  echo "  $n: heatmap@judge=${hm:-n/a} last: $last -> $verdict (criterion < $PASS_HM)"
done
