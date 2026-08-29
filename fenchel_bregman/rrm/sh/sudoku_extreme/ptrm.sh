#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"

: "${DATASET:?Set DATASET to the Sudoku-Extreme dataset directory}"
: "${CHECKPOINT:?Set CHECKPOINT to the Sudoku-Extreme PTRM checkpoint}"
OUTPUT_DIR="${OUTPUT_DIR:-${RRM_PROJECT_ROOT}/artifacts/sudoku_extreme/ptrm}"

run_table1_evaluation ptrm sudoku-extreme "${CHECKPOINT}" "${DATASET}" \
  "${OUTPUT_DIR}/evaluation" 10 64 1250 4 --latent-noise-scale 0.3
