#!/bin/bash
#SBATCH --job-name=eval-agi2-wptrm
#SBATCH --time=04:00:00
#SBATCH --gpus=4
#SBATCH --output=slurm/logs/%x-%j.out

# Evaluate W-PTRM (weight-space parameter perturbation) on ARC-AGI-2.
#
# Required: set CHECKPOINT to the trained ARC-AGI-2 checkpoint, e.g.
#   sbatch --export=ALL,CHECKPOINT=/path/to/step_N slurm/eval_agi2_w_ptrm.slurm
# Optional overrides (via --export): DATASET, OUTPUT_DIR, CANDIDATE_COUNT,
#   DEPTH, GLOBAL_BATCH_SIZE, PARAMETER_PERTURBATION_SCALE, SEED, MAX_BATCHES, CONFIG.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.slurm"

: "${CHECKPOINT:?Set CHECKPOINT to the ARC-AGI-2 PTRM checkpoint (sbatch --export=ALL,CHECKPOINT=...)}"

export DATASET="${DATASET:-${DATA_ROOT}/arc2concept-aug-1000}"
export OUTPUT_DIR="${OUTPUT_DIR:-${FB_DIR}/artifacts/agi-2/w_ptrm}"
export CANDIDATE_COUNT="${CANDIDATE_COUNT:-25}"
export DEPTH="${DEPTH:-16}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-768}"
export PARAMETER_PERTURBATION_SCALE="${PARAMETER_PERTURBATION_SCALE:-0.2}"
export SEED="${SEED:-0}"

bash rrm/sh/agi-2/w_ptrm.sh

echo "Job completed. Results in ${OUTPUT_DIR}"
