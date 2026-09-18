#!/usr/bin/env bash
# Configure the in-container conda environment, validate B200 CUDA, then run
# all 16 one-step smoke configurations. Run from the repository root.
set -euo pipefail

data_root="/data/nuscenes"
env_name="bevfusion-b200"
setup_args=()

usage() {
  echo "Usage: $0 [--data-root PATH] [--env-name NAME] [--use-system-cuda]"
}

while (($#)); do
  case "$1" in
    --data-root)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      data_root="$2"
      shift 2
      ;;
    --env-name)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      env_name="$2"
      shift 2
      ;;
    --use-system-cuda)
      setup_args+=("--use-system-cuda")
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
cd "${repo_root}"

echo "== host/container inventory"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
df -h "${repo_root}"
[[ -d "${data_root}" ]] || {
  echo "nuScenes root does not exist: ${data_root}" >&2
  exit 1
}
find "${data_root}" -maxdepth 1 -type d -print | sort

bash "${script_dir}/setup_b200_conda.sh" --env-name "${env_name}" "${setup_args[@]}"

echo "== strict CUDA environment check"
conda run --no-capture-output --name "${env_name}" \
  env PYTHONPATH="${repo_root}" \
  python docker/check_environment.py --strict --stack cu128 --cuda

echo "== 16 one-step smoke configurations"
conda run --no-capture-output --name "${env_name}" \
  env PYTHONPATH="${repo_root}" DATAROOT="${data_root}" DEVICE=0 \
  bash tools/ablation/smoke_all.sh

checkpoint="${repo_root}/pretrained/dsvt_nuscenes_official_lidar.pth"
if [[ -f "${checkpoint}" ]]; then
  echo "== full-model checkpoint load"
  conda run --no-capture-output --name "${env_name}" \
    env PYTHONPATH="${repo_root}" \
    python tools/dsvt_pretrained/check_full_model_load.py \
      configs/nuscenes/det/ablation/dsvt1_wf1_gf1_dal1.yaml \
      "${checkpoint}"
else
  echo "SKIP full-model load: checkpoint not found at ${checkpoint}" >&2
fi

echo "PASS: environment and 16 smoke configurations completed"
