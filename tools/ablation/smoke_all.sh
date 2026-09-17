#!/usr/bin/env bash
# Run on a CUDA server with mmcv, torchpack and nuScenes data installed.
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
dataroot="${DATAROOT:-${repo_root}/data/nuscenes}"
device="${DEVICE:-0}"
results="${repo_root}/tools/ablation/smoke_results.md"
configs=(
  dsvt0_wf0_gf0_dal0
  dsvt0_wf0_gf0_dal1
  dsvt0_wf0_gf1_dal0
  dsvt0_wf0_gf1_dal1
  dsvt0_wf1_gf0_dal0
  dsvt0_wf1_gf0_dal1
  dsvt0_wf1_gf1_dal0
  dsvt0_wf1_gf1_dal1
  dsvt1_wf0_gf0_dal0
  dsvt1_wf0_gf0_dal1
  dsvt1_wf0_gf1_dal0
  dsvt1_wf0_gf1_dal1
  dsvt1_wf1_gf0_dal0
  dsvt1_wf1_gf0_dal1
  dsvt1_wf1_gf1_dal0
  dsvt1_wf1_gf1_dal1
)

if [[ ! -f "${results}" ]]; then
  {
    echo "# Phase 2 server smoke results"
    echo
    echo "| UTC | config | result | smoke output |"
    echo "|---|---|---|---|"
  } >"${results}"
fi

failed=0
for name in "${configs[@]}"; do
  config="configs/nuscenes/det/ablation/${name}.yaml"
  log_file="$(mktemp "${TMPDIR:-/tmp}/bevfusion-smoke-${name}.XXXXXX")"
  if (
    cd "${repo_root}"
    python tools/smoke_train.py "${config}" --dataroot "${dataroot}" --device "${device}"
  ) 2>&1 | tee "${log_file}"; then
    status="PASS"
  else
    status="FAIL"
    failed=1
  fi
  summary="$(tail -n 1 "${log_file}" | tr '|' '/' | tr '\n' ' ')"
  printf '| %s | `%s` | %s | %s |\n' \
    "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "${config}" "${status}" "${summary}" \
    >>"${results}"
  rm -f "${log_file}"
done

exit "${failed}"
