#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"

: "${DATASET:?Set DATASET to the preprocessed ARC-AGI-1 dataset directory}"
: "${CHECKPOINT:?Set CHECKPOINT to the ARC-AGI-1 PTRM checkpoint}"
[[ -d "${DATASET}" ]] || { echo "Missing dataset: ${DATASET}" >&2; exit 2; }
[[ -f "${CHECKPOINT}" ]] || { echo "Missing checkpoint: ${CHECKPOINT}" >&2; exit 2; }

OUTPUT_DIR="${OUTPUT_DIR:-${RRM_PROJECT_ROOT}/artifacts/agi-1/w_ptrm}"
CANDIDATE_COUNT="${CANDIDATE_COUNT:-25}"
DEPTH="${DEPTH:-16}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-768}"
PARAMETER_PERTURBATION_SCALE="${PARAMETER_PERTURBATION_SCALE:-0.2}"
SEED="${SEED:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

EXTRA_ARGS=()
if [[ -n "${CONFIG:-}" ]]; then
  [[ -f "${CONFIG}" ]] || { echo "Missing config: ${CONFIG}" >&2; exit 2; }
  EXTRA_ARGS+=(--config "${CONFIG}")
fi
if [[ -n "${MAX_BATCHES:-}" ]]; then
  EXTRA_ARGS+=(--max-batches "${MAX_BATCHES}")
fi

COMMAND=(
  "${RRM_PYTHON_BIN}" -m torch.distributed.run
  --standalone "--nproc-per-node=${NPROC_PER_NODE}" --module rrm.arc
  --task arc-agi-1 --checkpoint "${CHECKPOINT}"
  --dataset "${DATASET}" --output "${OUTPUT_DIR}"
  --candidate-count "${CANDIDATE_COUNT}" --depth "${DEPTH}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}" --seed "${SEED}"
  --device "${RRM_DEVICE}"
  --parameter-perturbation-scale "${PARAMETER_PERTURBATION_SCALE}"
  "${EXTRA_ARGS[@]}"
)

if [[ -n "${GPU_IDS:-}" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${COMMAND[@]}"
else
  "${COMMAND[@]}"
fi
