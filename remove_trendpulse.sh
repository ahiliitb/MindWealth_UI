#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXIT_DIR="${SCRIPT_DIR}/chatbot/data/exit"

if [[ ! -d "${EXIT_DIR}" ]]; then
  printf 'Exit directory not found: %s\n' "${EXIT_DIR}" >&2
  exit 1
fi

cd "${EXIT_DIR}"

deleted_any=0

for asset_dir in */; do
  [[ -d "${asset_dir}" ]] || continue
  trendpulse_dir="${asset_dir%/}/TRENDPULSE"
  if [[ -d "${trendpulse_dir}" ]]; then
    printf 'Removing %s\n' "${trendpulse_dir}"
    rm -rf -- "${trendpulse_dir}"
    deleted_any=1
  fi
done

if [[ "${deleted_any}" -eq 0 ]]; then
  printf 'No TRENDPULSE directories found under %s\n' "${EXIT_DIR}"
else
  printf 'Completed removing TRENDPULSE directories.\n'
fi

