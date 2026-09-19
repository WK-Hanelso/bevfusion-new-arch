#!/usr/bin/env bash
# DSVT 학습 실패 이분 탐색 프로브 — A1 config(DSVT + DepthLSS + ConvFuser + TransFusion)로 3개 짧은 학습을 동시에.
#   P1 official_layout=true  + pretrained   (현재 설정, 대조군)
#   P2 official_layout=true  + scratch      (사전학습 제거)
#   P3 official_layout=false + scratch      (레거시 A6000 성공 조건)
# 6,000샘플 서브셋(≈1,900 iter, 배치 16×누적 2), 평가 없음. 60초마다 loss_heatmap/matched_ious를 화면에 찍는다.
# 사용: bash tools/ablation/probe_dsvt.sh <nuscenes_root> [GPU1 GPU2 GPU3]   (기본 1 5 6)
set -uo pipefail
ROOT="${1:?nuscenes_root}"; G1="${2:-1}"; G2="${3:-5}"; G3="${4:-6}"
cd "$(dirname "$0")/../.."
ENV=bevfusion-b200; N="${PROBE_TRAIN_SAMPLES:-6000}"; CFG=configs/nuscenes/det/ablation/dsvt1_wf0_gf0_dal0.yaml
PR=experiments/probe; MINI="$PR/mini_nuscenes"; mkdir -p "$MINI"
for d in samples sweeps maps v1.0-trainval; do [ -e "$MINI/$d" ] || ln -s "$ROOT/$d" "$MINI/$d"; done
[ -e "$MINI/nuscenes_infos_val.pkl" ] || ln -s "$ROOT/nuscenes_infos_val.pkl" "$MINI/nuscenes_infos_val.pkl"
conda run -n "$ENV" --no-capture-output python - "$ROOT" "$MINI" "$N" <<'PY'
import pickle, sys
root, mini, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = pickle.load(open(f"{root}/nuscenes_infos_train.pkl", "rb")); d["infos"] = d["infos"][:n]
pickle.dump(d, open(f"{mini}/nuscenes_infos_train.pkl", "wb")); print("probe train infos:", len(d["infos"]))
PY
trap 'echo "== stop"; pkill -TERM -f "experiments/probe/P" 2>/dev/null; sleep 3; pkill -KILL -f "experiments/probe/P" 2>/dev/null; exit 130' INT TERM
run() { # name gpu port extra...
  local NAME="$1" GPU="$2" PORT="$3"; shift 3; local RUN="$PR/$NAME"; rm -rf "$RUN"; mkdir -p "$RUN"
  CUDA_VISIBLE_DEVICES="$GPU" conda run -n "$ENV" --no-capture-output torchrun --master_port="$PORT" --nproc_per_node=1 \
    tools/train_torchrun.py "$CFG" --run-dir "$RUN" --max_epochs 1 --dataset_root "$MINI/" --seed 0 --fp16 None \
    --find_unused_parameters True --checkpoint_config.out_dir "$RUN/checkpoints" --checkpoint_config.interval 99 --evaluation.interval 99 \
    --data.samples_per_gpu 16 --data.workers_per_gpu 8 --optimizer_config.type GradientCumulativeOptimizerHook --optimizer_config.cumulative_iters 2 \
    "$@" > "$RUN/train.log" 2>&1 &
}
echo "== probe start $(date)  gpus=$G1,$G2,$G3  samples=$N"
run P1_official_pretrained "$G1" 29801 --load_from pretrained/dsvt_nuscenes_official_lidar.pth
run P2_official_scratch    "$G2" 29802
run P3_legacy_scratch      "$G3" 29803 --model.encoders.lidar.official_layout False
T0=$(date +%s)
while pgrep -f "experiments/probe/P" >/dev/null; do
  sleep 60; s=$(( $(date +%s) - T0 )); printf -- '---- [%02d:%02d 경과] ----\n' $((s/60)) $((s%60))
  for d in "$PR"/P*/; do
    n=$(basename "$d")
    line=$(grep -hE 'Epoch \[' "$d/train.log" 2>/dev/null | tail -1 | grep -oE "Epoch \[1\]\[[0-9]+/[0-9]+\]|loss_heatmap: [0-9.]+|loss_bbox: [0-9.]+|matched_ious: [0-9.]+|grad_norm: [0-9.]+" | tr '\n' ' ')
    err=$(grep -hE "Error" "$d/train.log" 2>/dev/null | grep -v Warning | tail -1 | cut -c1-80)
    printf '  %-24s %s %s\n' "$n" "$line" "${err:+ERR: $err}"
  done
done
echo "== probe done $(date)"
for d in "$PR"/P*/; do echo "== $(basename "$d")"; grep -hE 'Epoch \[1\]\[(50|500|1000|1500|1900)/' "$d/train.log" | grep -oE "Epoch \[1\]\[[0-9]+|loss_heatmap: [0-9.]+|matched_ious: [0-9.]+" | paste -d' ' - - -; done
