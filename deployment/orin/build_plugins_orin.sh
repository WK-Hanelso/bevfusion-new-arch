#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/../.." && pwd)"
source_dir="${repo_dir}/deployment/tensorrt/plugins"
build_dir="${BUILD_DIR:-${script_dir}/build/plugins-trt85}"
install_dir="${INSTALL_DIR:-${script_dir}/build/plugins-trt85-install}"

cmake -S "${source_dir}" -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DTHOR_CUDA_ARCHITECTURES=87 \
  -DDYNAMIC_BEVFUSION_MAX_POINTS=100000 \
  -DDYNAMIC_BEVFUSION_MAX_PILLARS=10000 \
  -DDYNAMIC_BEVFUSION_MAX_SETS=512
cmake --build "${build_dir}" --parallel "$(nproc)"
cmake --install "${build_dir}" --prefix "${install_dir}"

plugins=(
  libdynamic_pillar_decorate.so
  libdynamic_scatter_max.so
  libtensor_barrier.so
  libdsvt_rotated_set.so
  libdsvt_dense_scatter.so
)
for plugin in "${plugins[@]}"; do
  test -s "${install_dir}/${plugin}"
done

echo "PASS Orin TensorRT plugins: ${install_dir}"
