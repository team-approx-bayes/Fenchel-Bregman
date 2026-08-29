#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"

: "${DATASET:?Set DATASET to the Maze-Hard dataset directory}"
OUTPUT_DIR="${OUTPUT_DIR:-${RRM_PROJECT_ROOT}/artifacts/maze_hard/gram}"
CHECKPOINT="${CHECKPOINT:-}"

if [[ -z "${CHECKPOINT}" ]]; then
  : "${BASE_CHECKPOINT:?Set BASE_CHECKPOINT to the Maze-Hard PTRM checkpoint}"
  "${RRM_PYTHON_BIN}" -m rrm.train \
    --model gram --task maze-hard --preset paper \
    --checkpoint "${BASE_CHECKPOINT}" --dataset "${DATASET}" \
    --output "${OUTPUT_DIR}/training" --device "${RRM_DEVICE}"
  CHECKPOINT="${OUTPUT_DIR}/training/checkpoint.pt"
fi

run_table1_evaluation gram maze-hard "${CHECKPOINT}" "${DATASET}" \
  "${OUTPUT_DIR}/evaluation" 10 16 32 1
