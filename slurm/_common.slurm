#!/bin/bash
# Shared environment setup for rikyu cluster SLURM jobs (posterior-decision-reasoning).
# Sourced by the per-job .slurm scripts; not meant to be submitted directly.

# 1. Proxy for internet access on compute nodes (needed for wandb sync / HF hub)
export HTTP_PROXY="http://c000:8888"
export HTTPS_PROXY="http://c000:8888"
export http_proxy="http://c000:8888"
export https_proxy="http://c000:8888"
export NO_PROXY="localhost,127.0.0.1,c000,c*,*.rikyu.r-ccs.riken.jp,10.134.0.0/16"
export no_proxy="localhost,127.0.0.1,c000,c*,*.rikyu.r-ccs.riken.jp,10.134.0.0/16"

# 2. Single-node distributed settings (torch.distributed.run --standalone)
export MASTER_ADDR="127.0.0.1"
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"

# 3. CUDA toolkit (torch cu130 build; GB200 / sm_100 GPUs)
module load "${CUDA_MODULE:-cuda/13.2.2}"

# 4. Allocator tuning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 5. Project paths
PROJECT_ROOT="${PROJECT_ROOT:-/data1/rkp00055/adrian/projects/posterior-decision-reasoning}"
FB_DIR="${PROJECT_ROOT}/fenchel_bregman"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"

# 6. Local overrides (wandb project, etc.) from the repo-root .env, if present.
#    Values already set in the environment take precedence over .env.
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  while IFS='=' read -r key value; do
    # skip comments and blank lines
    [[ "${key}" =~ ^[[:space:]]*# ]] && continue
    [[ -z "${key// }" ]] && continue
    # only set if not already present in the environment
    if [[ -z "${!key+x}" ]]; then
      export "${key}=${value}"
    fi
  done < "${PROJECT_ROOT}/.env"
fi

# Use the fenchel_bregman uv venv python
export PYTHON_BIN="${PYTHON_BIN:-${FB_DIR}/.venv/bin/python}"

# Number of GPUs requested via #SBATCH --gpus=N drives nproc-per-node
NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS:-1}}"
export NPROC_PER_NODE

cd "${FB_DIR}"
