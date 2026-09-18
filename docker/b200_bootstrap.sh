#!/usr/bin/env bash
# B200 호스트 1차 점검 + cu128 학습 이미지 빌드 + 16개 config smoke. 저장소 루트에서 실행.
# 사용: bash docker/b200_bootstrap.sh [nuscenes_root]   (기본: /data/nuscenes)
set -uo pipefail
ROOT="${1:-/data/nuscenes}"
echo "== host"; hostname; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -8
docker --version; docker info 2>/dev/null | grep -i "runtimes" ; df -h "$PWD" | tail -1
echo "== data root: $ROOT"; ls "$ROOT" 2>/dev/null | head; ls "$ROOT"/v1.0-trainval 2>/dev/null | head -3; ls "$ROOT"/nuscenes_infos_train.pkl "$ROOT"/nuscenes_infos_val.pkl 2>/dev/null
echo "== build cu128 image"
docker build --platform linux/amd64 -f docker/Dockerfile.cu128 -t bevfusion-train:cu128 . 2>&1 | tail -5 || exit 1
echo "== environment check"
docker run --rm --gpus all --user "$(id -u):$(id -g)" -v "$PWD/.cache/bevfusion:/workspace/cache" -e HOME=/workspace/cache -w /workspace/bevfusion bevfusion-train:cu128 \
  python docker/check_environment.py --strict 2>&1 | tail -5
echo "== smoke 16 configs (mini 또는 full infos 사용)"
mkdir -p runs .cache/bevfusion experiments
docker run --rm --gpus all --shm-size=16g --user "$(id -u):$(id -g)" \
  -v "$ROOT:/workspace/bevfusion/data/nuscenes:ro" -v "$PWD/configs:/workspace/bevfusion/configs:ro" -v "$PWD/tools:/workspace/bevfusion/tools" \
  -v "$PWD/pretrained:/workspace/bevfusion/pretrained:ro" -v "$PWD/runs:/workspace/bevfusion/runs" -v "$PWD/experiments:/workspace/bevfusion/experiments" \
  -v "$PWD/.cache/bevfusion:/workspace/cache" -e HOME=/workspace/cache -e DEVICE=0 -e PYTHONPATH=/workspace/bevfusion \
  -w /workspace/bevfusion bevfusion-train:cu128 bash -lc "bash tools/ablation/smoke_all.sh" 2>&1 | tail -20
echo "== smoke results"; cat tools/ablation/smoke_results.md 2>/dev/null | tail -18
