#!/usr/bin/env bash
# 노트북(2060) 컨테이너 실행 래퍼. 이미지 안의 mmdet3d(CUDA 확장 포함)를 쓰고, 호스트에서는
# configs/tools/pretrained/runs/experiments/deployment 만 마운트한다. 사용: bash tools/ablation/run_smoke_local.sh <컨테이너 내 명령...>
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p runs .cache/bevfusion experiments
exec docker run --rm --gpus all --shm-size=8g --user "$(id -u):$(id -g)" \
  -v /mnt/hdd/nuscenes:/workspace/bevfusion/data/nuscenes:ro \
  -v "$PWD/configs:/workspace/bevfusion/configs:ro" \
  -v "$PWD/tools:/workspace/bevfusion/tools" \
  -v "$PWD/pretrained:/workspace/bevfusion/pretrained:ro" \
  -v "$PWD/runs:/workspace/bevfusion/runs" \
  -v "$PWD/experiments:/workspace/bevfusion/experiments" \
  -v "$PWD/deployment:/workspace/bevfusion/deployment" \
  -v "$PWD/.cache/bevfusion:/workspace/cache" \
  -e HOME=/workspace/cache -e DEVICE=0 -e PYTHONPATH=/workspace/bevfusion \
  -w /workspace/bevfusion bevfusion-train:cu113 \
  bash -lc "$*"
