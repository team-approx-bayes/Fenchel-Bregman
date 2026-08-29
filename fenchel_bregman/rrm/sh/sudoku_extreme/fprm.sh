#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"

: "${DATASET:?Set DATASET to the Sudoku-Extreme dataset directory}"
: "${CHECKPOINT:?Set CHECKPOINT to the released Sudoku-Extreme FPRM checkpoint}"
OUTPUT_DIR="${OUTPUT_DIR:-${RRM_PROJECT_ROOT}/artifacts/sudoku_extreme/fprm}"

run_table1_evaluation fprm sudoku-extreme "${CHECKPOINT}" "${DATASET}" \
  "${OUTPUT_DIR}/evaluation" 1 64 768 1
