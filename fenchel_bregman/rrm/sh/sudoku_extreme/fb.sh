#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"

: "${DATASET:?Set DATASET to the Sudoku-Extreme dataset directory}"
OUTPUT_DIR="${OUTPUT_DIR:-${RRM_PROJECT_ROOT}/artifacts/sudoku_extreme/fb}"
CHECKPOINT="${CHECKPOINT:-}"

if [[ -z "${CHECKPOINT}" ]]; then
  : "${BASE_CHECKPOINT:?Set BASE_CHECKPOINT to the Sudoku-Extreme PTRM checkpoint}"
  "${RRM_PYTHON_BIN}" -m rrm.train \
    --model ptrm --task sudoku-extreme --preset paper --fb \
    --checkpoint "${BASE_CHECKPOINT}" --dataset "${DATASET}" \
    --output "${OUTPUT_DIR}/training" --device "${RRM_DEVICE}"
  CHECKPOINT="${OUTPUT_DIR}/training/checkpoint.pt"
fi

run_table1_evaluation ptrm sudoku-extreme "${CHECKPOINT}" "${DATASET}" \
  "${OUTPUT_DIR}/evaluation" 10 64 64 8 --fb
