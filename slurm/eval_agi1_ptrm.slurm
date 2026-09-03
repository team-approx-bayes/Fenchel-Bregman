#!/bin/bash
#SBATCH --job-name=eval-agi1-ptrm
#SBATCH --time=04:00:00
#SBATCH --gpus=4
#SBATCH --output=slurm/logs/%x-%j.out

# Evaluate PTRM (latent-noise posterior sampling) on ARC-AGI-1.
#
# Required: set CHECKPOINT to the trained ARC-AGI-1 checkpoint, e.g.
#   sbatch --export=ALL,CHECKPOINT=/path/to/step_N slurm/eval_agi1_ptrm.slurm
# Optional overrides (via --export): DATASET, OUTPUT_DIR, CANDIDATE_COUNT,
#   DEPTH, GLOBAL_BATCH_SIZE, LATENT_NOISE_SCALE, SEED, MAX_BATCHES, CONFIG.

# sbatch spools this script to /var/spool/slurmd, so BASH_SOURCE cannot be used
# to locate _common.slurm; use the absolute project path instead.
PROJECT_ROOT="${PROJECT_ROOT:-/data1/rkp00055/adrian/projects/posterior-decision-reasoning}"
source "${PROJECT_ROOT}/slurm/_common.slurm"

: "${CHECKPOINT:?Set CHECKPOINT to the ARC-AGI-1 PTRM checkpoint (sbatch --export=ALL,CHECKPOINT=...)}"

export DATASET="${DATASET:-${DATA_ROOT}/arc1concept-aug-1000}"
export OUTPUT_DIR="${OUTPUT_DIR:-${FB_DIR}/artifacts/agi-1/ptrm}"
export CANDIDATE_COUNT="${CANDIDATE_COUNT:-25}"
export DEPTH="${DEPTH:-16}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
export LATENT_NOISE_SCALE="${LATENT_NOISE_SCALE:-0.2}"
export SEED="${SEED:-0}"

bash rrm/sh/agi-1/ptrm.sh

echo "Job completed. Results in ${OUTPUT_DIR}"
