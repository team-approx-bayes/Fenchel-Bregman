#!/usr/bin/env bash
# Download the official TRM checkpoints from the ARC Prize verification repo
# (https://huggingface.co/arcprize/trm_arc_prize_verification) into MODELS_ROOT.
#
# These are the reference TRM checkpoints for ARC-AGI-1 (arc_v1_public) and
# ARC-AGI-2 (arc_v2_public); each directory contains the step_N checkpoint plus
# the all_config.yaml needed by fenchel_bregman's rrm.arc evaluator.
#
# Usage: download_trm_checkpoints.sh [MODELS_ROOT]
#   MODELS_ROOT defaults to ${ROOT}/models.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
require_trm_python

MODELS_ROOT="${1:-${MODELS_ROOT:-${ROOT}/models}}"
HF_REPO="arcprize/trm_arc_prize_verification"
mkdir -p "${MODELS_ROOT}"

download_dir() {
  local subdir="$1" dest="${MODELS_ROOT}/$1"
  if [[ -d "${dest}" ]] && compgen -G "${dest}/step_*" >/dev/null; then
    info "Already present: ${dest} (skipping; delete to re-download)"
    return 0
  fi
  info "Downloading ${HF_REPO}/${subdir} -> ${dest}"
  "${TRM_PYTHON}" - "${HF_REPO}" "${subdir}" "${MODELS_ROOT}" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo, subdir, dest = sys.argv[1], sys.argv[2], sys.argv[3]
path = snapshot_download(
    repo_id=repo,
    repo_type="model",
    allow_patterns=[f"{subdir}/*"],
    local_dir=dest,
)
print(f"downloaded to {path}")
PY
}

download_dir arc_v1_public
download_dir arc_v2_public

ok "TRM checkpoints under ${MODELS_ROOT}:"
du -sh "${MODELS_ROOT}"/arc_v1_public "${MODELS_ROOT}"/arc_v2_public 2>/dev/null || true
