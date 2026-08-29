#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"

: "${DATASET:?Set DATASET to the Maze-Hard dataset directory}"
OUTPUT_DIR="${OUTPUT_DIR:-${RRM_PROJECT_ROOT}/artifacts/maze_hard/fb_fprm}"
CHECKPOINT="${CHECKPOINT:-}"

if [[ -z "${CHECKPOINT}" ]]; then
  : "${BASE_CHECKPOINT:?Set BASE_CHECKPOINT to the reconstructed FPRM IVON initializer}"
  "${RRM_PYTHON_BIN}" -m torch.distributed.run --standalone \
    --nproc-per-node "${NPROC_PER_NODE:-8}" --module rrm.train \
    --model fprm --task maze-hard --preset paper --fb \
    --checkpoint "${BASE_CHECKPOINT}" --dataset "${DATASET}" \
    --output "${OUTPUT_DIR}/training" --device "${RRM_DEVICE}"
  CHECKPOINT="${OUTPUT_DIR}/training/checkpoint.pt"
fi

run_table1_evaluation fprm maze-hard "${CHECKPOINT}" "${DATASET}" \
  "${OUTPUT_DIR}/evaluation" 10 16 100 1 --fb
