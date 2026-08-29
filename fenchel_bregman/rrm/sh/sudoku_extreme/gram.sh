#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"

: "${DATASET:?Set DATASET to the Sudoku-Extreme dataset directory}"
OUTPUT_DIR="${OUTPUT_DIR:-${RRM_PROJECT_ROOT}/artifacts/sudoku_extreme/gram}"
CHECKPOINT="${CHECKPOINT:-}"

if [[ -z "${CHECKPOINT}" ]]; then
  "${RRM_PYTHON_BIN}" -m rrm.train \
    --model gram --task sudoku-extreme --preset paper \
    --dataset "${DATASET}" --output "${OUTPUT_DIR}/training" \
    --device "${RRM_DEVICE}"
  CHECKPOINT="${OUTPUT_DIR}/training/checkpoint.pt"
fi

run_table1_evaluation gram sudoku-extreme "${CHECKPOINT}" "${DATASET}" \
  "${OUTPUT_DIR}/evaluation" 10 64 16 8
