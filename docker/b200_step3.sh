#!/usr/bin/env bash
# B200 D0 step 3 — 압축 해제 완료 후 실행. infos 생성 → 16 config smoke → 속도 실측(c4_full, 4 GPU, 300 iter).
# 사용: bash docker/b200_step3.sh <nuscenes_root> [smoke|speed|all]   (기본 all)
set -uo pipefail
ROOT="${1:?nuscenes_root}"; MODE="${2:-all}"; cd "$(dirname "$0")/.."
ENV=bevfusion-b200; export PYTHONPATH="$PWD"
echo "== step3 start $(date) root=$ROOT mode=$MODE"
mkdir -p data && [ -e data/nuscenes ] || ln -s "$ROOT" data/nuscenes
echo "== 0) data sanity"; ls data/nuscenes/v1.0-trainval | head -3; ls data/nuscenes/samples | wc -l; ls data/nuscenes/sweeps | wc -l; ls data/nuscenes/maps | head -2
if [ ! -f data/nuscenes/nuscenes_infos_train.pkl ]; then
  echo "== 1) create infos (v1.0-trainval, max_sweeps=10) $(date)"
  conda run -n $ENV --no-capture-output python -c "from tools.data_converter.nuscenes_converter import create_nuscenes_infos; create_nuscenes_infos('data/nuscenes', 'nuscenes', version='v1.0-trainval', max_sweeps=10)" 2>&1 | tail -5
  ls -la data/nuscenes/nuscenes_infos_train.pkl data/nuscenes/nuscenes_infos_val.pkl
else echo "== 1) infos exist"; fi
conda run -n $ENV --no-capture-output python - <<'PY'
import pickle
for f in ["data/nuscenes/nuscenes_infos_train.pkl","data/nuscenes/nuscenes_infos_val.pkl"]:
    d=pickle.load(open(f,"rb")); print(f, "n=", len(d["infos"]), d.get("metadata"))
PY
if [ "$MODE" = "all" ] || [ "$MODE" = "smoke" ]; then
  echo "== 2) smoke 16 configs (GPU 0) $(date)"
  rm -f tools/ablation/smoke_results.md
  DEVICE=0 conda run -n $ENV --no-capture-output bash tools/ablation/smoke_all.sh 2>&1 | tail -5
  cat tools/ablation/smoke_results.md | tail -18
fi
if [ "$MODE" = "all" ] || [ "$MODE" = "speed" ]; then
  echo "== 3) speed probe: c4_full (dsvt1_wf1_gf1_dal1) 4 GPU, DSVT init, 300 iter $(date)"
  mkdir -p runs
  CUDA_VISIBLE_DEVICES=0,1,2,3 conda run -n $ENV --no-capture-output torchrun --nproc_per_node=4 tools/train_torchrun.py \
    configs/nuscenes/det/ablation/dsvt1_wf1_gf1_dal1.yaml --run-dir runs/speed_c4_full_4gpu --find_unused_parameters True --fp16 None \
    --load_from pretrained/dsvt_nuscenes_official_lidar.pth 2>&1 | grep -E "Epoch \[|time:|memory:|Error|error" | head -40 &
  TP=$!; sleep 900; kill $TP 2>/dev/null; sleep 5; pkill -f "tools/train_torchrun.py" 2>/dev/null
  echo "-- last log lines"; L=$(ls -t runs/speed_c4_full_4gpu/*.log 2>/dev/null | head -1); [ -n "$L" ] && grep -E "Epoch \[" "$L" | tail -5
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -4
fi
echo "== step3 done $(date)"
