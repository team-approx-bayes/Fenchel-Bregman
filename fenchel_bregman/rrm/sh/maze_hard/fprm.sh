#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"

: "${DATASET:?Set DATASET to the Maze-Hard dataset directory}"
: "${CHECKPOINT:?Set CHECKPOINT to the released Maze-Hard FPRM checkpoint}"
OUTPUT_DIR="${OUTPUT_DIR:-${RRM_PROJECT_ROOT}/artifacts/maze_hard/fprm}"

run_table1_evaluation fprm maze-hard "${CHECKPOINT}" "${DATASET}" \
  "${OUTPUT_DIR}/evaluation" 1 16 768 1
