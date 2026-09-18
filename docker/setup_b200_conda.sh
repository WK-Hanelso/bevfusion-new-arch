#!/usr/bin/env bash
# Idempotently create/update the B200 training environment inside an existing
# GPU container. Run from anywhere in the repository checkout.
set -euo pipefail

env_name="bevfusion-b200"
force_system_cuda=0

usage() {
  echo "Usage: $0 [--env-name NAME] [--use-system-cuda]"
}

while (($#)); do
  case "$1" in
    --env-name)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      env_name="$2"
      shift 2
      ;;
    --use-system-cuda)
      force_system_cuda=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
env_file="${script_dir}/b200_conda_env.yml"
requirements_file="${script_dir}/requirements-cu128.txt"

command -v conda >/dev/null 2>&1 || {
  echo "conda is required in the parent B200 container" >&2
  exit 1
}

echo "== GPU inventory"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
else
  echo "WARNING: nvidia-smi is unavailable; GPU validation will fail later" >&2
fi

cuda_release() {
  "$1" --version 2>/dev/null \
    | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' \
    | head -n 1
}

cuda_12_8_or_newer_minor() {
  local release="$1"
  local major="${release%%.*}"
  local minor="${release#*.}"
  [[ "${major}" == "12" && "${minor}" =~ ^[0-9]+$ && "${minor}" -ge 8 ]]
}

system_nvcc="$(command -v nvcc || true)"
system_release=""
if [[ -n "${system_nvcc}" ]]; then
  system_release="$(cuda_release "${system_nvcc}")"
  echo "== system nvcc: ${system_nvcc} (CUDA ${system_release:-unknown})"
else
  echo "== system nvcc: not found"
fi

use_system_cuda=0
if [[ -n "${system_release}" ]] && cuda_12_8_or_newer_minor "${system_release}"; then
  use_system_cuda=1
fi
if ((force_system_cuda)) && ((use_system_cuda == 0)); then
  echo "--use-system-cuda requires a CUDA 12.8.x or 12.9.x nvcc" >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -Fxq "${env_name}"; then
  echo "== update conda environment: ${env_name}"
  conda env update --name "${env_name}" --file "${env_file}"
else
  echo "== create conda environment: ${env_name}"
  conda env create --name "${env_name}" --file "${env_file}"
fi

if ((use_system_cuda)); then
  cuda_home="$(cd -- "$(dirname -- "${system_nvcc}")/.." && pwd)"
  echo "== use system CUDA ${system_release}: ${cuda_home}"
else
  echo "== install/confirm conda CUDA toolkit 12.8"
  conda install --name "${env_name}" --channel nvidia --yes "cuda-toolkit=12.8"
  env_prefix="$(conda run --no-capture-output --name "${env_name}" python -c 'import sys; print(sys.prefix)')"
  cuda_home="${env_prefix}"
fi

run_in_env() {
  conda run --no-capture-output --name "${env_name}" \
    env CUDA_HOME="${cuda_home}" \
        TORCH_CUDA_ARCH_LIST="9.0;10.0" \
        BEVFUSION_CUDA_ARCHS="90;100" \
        MAX_JOBS="${MAX_JOBS:-4}" \
        "$@"
}

echo "== install official torch cu128 wheels"
run_in_env python -m pip install --no-cache-dir \
  torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128

echo "== install Python runtime dependencies"
run_in_env python -m pip install --no-cache-dir --requirement "${requirements_file}"

echo "== build/install mmcv-full and torch-scatter"
run_in_env env MMCV_WITH_OPS=1 FORCE_CUDA=1 \
  python -m pip install --no-cache-dir --force-reinstall --no-build-isolation --no-deps \
  --no-binary=mmcv-full mmcv-full==1.7.2
run_in_env env FORCE_CUDA=1 \
  python -m pip install --no-cache-dir --force-reinstall --no-build-isolation --no-deps \
  --no-binary=torch-scatter torch-scatter==2.1.2

echo "== build/install repository extensions"
run_in_env env FORCE_CUDA=1 \
  python -m pip install --no-cache-dir --editable "${repo_root}" \
  --no-deps --no-build-isolation

echo "== validate installed environment"
run_in_env python -m pip check
run_in_env python "${script_dir}/check_environment.py" --strict --stack cu128

echo "PASS: conda environment ${env_name} is ready"
echo "CUDA_HOME=${cuda_home}"
echo "Run GPU validation with: conda run -n ${env_name} python docker/check_environment.py --strict --stack cu128 --cuda"
