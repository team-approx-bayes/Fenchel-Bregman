#!/usr/bin/env bash
# Build the Sudoku-Extreme and Maze-Hard datasets (Table 1 tasks).
# These download their raw data from Hugging Face on first run.
#
# Usage: build_table1_datasets.sh [DATA_ROOT]
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
require_trm_python

DATA_ROOT="${1:-${DATA_ROOT:-${ROOT}/data}}"
mkdir -p "${DATA_ROOT}"

info "Building Sudoku-Extreme (1000 examples, 1000 augments) -> ${DATA_ROOT}/sudoku-extreme-1k-aug-1000"
(cd "${TRM_DIR}" && "${TRM_PYTHON}" dataset/build_sudoku_dataset.py \
  --output-dir "${DATA_ROOT}/sudoku-extreme-1k-aug-1000" \
  --subsample-size 1000 --num-aug 1000)

info "Building Maze-Hard (1000 examples, 8 augments) -> ${DATA_ROOT}/maze-30x30-hard-1k"
(cd "${TRM_DIR}" && "${TRM_PYTHON}" dataset/build_maze_dataset.py \
  --output-dir "${DATA_ROOT}/maze-30x30-hard-1k")

ok "Table 1 datasets built under ${DATA_ROOT}:"
du -sh "${DATA_ROOT}"/sudoku-extreme-1k-aug-1000 "${DATA_ROOT}"/maze-30x30-hard-1k 2>/dev/null || true
