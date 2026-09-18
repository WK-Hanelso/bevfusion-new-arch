#!/usr/bin/env bash
# B200 D0 step 2 — 저장소 체크아웃 안에서 실행. conda 환경 구축 + DSVT 공식 ckpt 변환 + 데이터 경로 점검.
# 사용: bash docker/b200_step2.sh <nuscenes_root>    (예: ~/culee/data/nuscenes)
set -uo pipefail
ROOT="${1:-}"; cd "$(dirname "$0")/.."
echo "== step2 start $(date)"; echo "repo: $(git rev-parse --short HEAD) $(git branch --show-current)"
echo "== 1) conda env (cuda-toolkit 12.8 in env; system nvcc 13.0 is NOT used)"
MAX_JOBS="${MAX_JOBS:-32}" bash docker/setup_b200_conda.sh --env-name bevfusion-b200 2>&1 | tee docker/b200_setup.log | tail -25
echo "== 2) environment check"
conda run -n bevfusion-b200 python docker/check_environment.py --strict 2>&1 | tail -8
conda run -n bevfusion-b200 python -c "import torch, mmcv, mmdet3d; print('torch', torch.__version__, torch.version.cuda, 'gpus', torch.cuda.device_count(), torch.cuda.get_device_name(0)); print('mmcv', mmcv.__version__)"
echo "== 3) DSVT official checkpoint -> converted"
mkdir -p pretrained
if [ ! -f pretrained/DSVT_Nuscenes_val.pth ]; then conda run -n bevfusion-b200 pip install -q gdown && conda run -n bevfusion-b200 python -m gdown 10d7c-uJxg5w4GN-JmRBQi4gQDwHiOHxP -O pretrained/DSVT_Nuscenes_val.pth; fi
sha256sum pretrained/DSVT_Nuscenes_val.pth | cut -c1-16
conda run -n bevfusion-b200 python tools/dsvt_pretrained/convert_official_dsvt.py pretrained/DSVT_Nuscenes_val.pth --output pretrained/dsvt_nuscenes_official_lidar.pth 2>&1 | tail -4 || echo "convert: 인자 확인 필요 (python tools/dsvt_pretrained/convert_official_dsvt.py --help)"
echo "== 4) data root: ${ROOT:-<미지정>}"
if [ -n "$ROOT" ] && [ -d "$ROOT" ]; then ls "$ROOT" | head -12; ls "$ROOT"/v1.0-trainval 2>/dev/null | head -3; du -sh "$ROOT"/samples "$ROOT"/sweeps 2>/dev/null; ls "$ROOT"/nuscenes_infos_train.pkl "$ROOT"/nuscenes_infos_val.pkl 2>/dev/null || echo "infos pkl 없음 → step3에서 생성"; fi
echo "== step2 done $(date)"
