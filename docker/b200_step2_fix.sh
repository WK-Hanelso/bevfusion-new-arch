#!/usr/bin/env bash
# B200 step2 보정 — step2에서 실패한 두 항목만 재실행.
#  (a) 저장소 CUDA 확장 editable 설치: setuptools 80은 pyproject.toml 없는 프로젝트의 develop을
#      `pip install -e . --use-pep517`(빌드 격리 ON)로 위임 → 격리 env에 torch 없음 → 실패. setuptools<80으로 고정 후 재설치.
#  (b) DSVT 공식 ckpt: gdown --fuzzy 미지원 버전 → 파일 id 직접 지정.
# 사용: bash docker/b200_step2_fix.sh
set -uo pipefail
cd "$(dirname "$0")/.."
ENV=bevfusion-b200
echo "== fix start $(date)  repo: $(git rev-parse --short HEAD)"
echo "== a1) pin setuptools<80"
conda run -n $ENV python -m pip install --no-cache-dir "setuptools==69.5.1" 2>&1 | tail -1
conda run -n $ENV python -c "import setuptools; print('setuptools', setuptools.__version__)"
echo "== a2) rebuild repository CUDA extensions (editable)"
conda run -n $ENV env CUDA_HOME="$HOME/.conda/envs/$ENV" TORCH_CUDA_ARCH_LIST="9.0;10.0" BEVFUSION_CUDA_ARCHS="90;100" MAX_JOBS="${MAX_JOBS:-32}" FORCE_CUDA=1 \
  python -m pip install --no-cache-dir --editable . --no-deps --no-build-isolation 2>&1 | tee docker/b200_fix_build.log | grep -E "error|Error|Successfully|Finished" | tail -5
echo "== a3) ops import check"
ls mmdet3d/ops/*/*.so 2>/dev/null | head -20
conda run -n $ENV python -c "
import mmdet3d, mmdet3d.ops as ops
from mmdet3d.ops import bev_pool, Voxelization, spconv
print('mmdet3d ops OK', mmdet3d.__file__)"
echo "== b1) DSVT official checkpoint (sha256 prefix expected a675149d095eef8d)"
mkdir -p pretrained
if [ ! -s pretrained/DSVT_Nuscenes_val.pth ]; then
  conda run -n $ENV python -m gdown 10d7c-uJxg5w4GN-JmRBQi4gQDwHiOHxP -O pretrained/DSVT_Nuscenes_val.pth 2>&1 | tail -2 \
  || conda run -n $ENV python -m gdown "https://drive.google.com/uc?id=10d7c-uJxg5w4GN-JmRBQi4gQDwHiOHxP" -O pretrained/DSVT_Nuscenes_val.pth 2>&1 | tail -2
fi
ls -la pretrained/DSVT_Nuscenes_val.pth && sha256sum pretrained/DSVT_Nuscenes_val.pth | cut -c1-16
echo "== b2) convert"
conda run -n $ENV python tools/dsvt_pretrained/convert_official_dsvt.py pretrained/DSVT_Nuscenes_val.pth --output pretrained/dsvt_nuscenes_official_lidar.pth 2>&1 | tail -4
ls -la pretrained/dsvt_nuscenes_official_lidar.pth
echo "== c) data extraction status"
pgrep -fa "[t]ar " | cut -c1-100; du -sh "${1:-$HOME/culee/data/nuscenes}"/samples "${1:-$HOME/culee/data/nuscenes}"/sweeps 2>/dev/null
echo "== fix done $(date)"
