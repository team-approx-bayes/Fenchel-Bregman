#!/usr/bin/env bash
# Clone or update the code repositories used by this project.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

TRM_REPO="${TRM_REPO:-https://github.com/SamsungSAILMontreal/TinyRecursiveModels.git}"
FB_REPO="${FB_REPO:-https://github.com/team-approx-bayes/posterior-decision-reasoning.git}"

clone_or_update() {
  local dir="$1" repo="$2"
  if [[ -d "${dir}/.git" ]]; then
    info "Updating existing repo in ${dir}"
    git -C "${dir}" fetch --all --prune
    git -C "${dir}" pull --ff-only || warn "pull failed for ${dir}; leaving working tree as-is"
  elif [[ -d "${dir}" ]]; then
    warn "${dir} exists but is not a git repo; skipping clone"
  else
    info "Cloning ${repo} into ${dir}"
    git clone "${repo}" "${dir}"
  fi
}

clone_or_update "${TRM_DIR}" "${TRM_REPO}"
clone_or_update "${FB_DIR}" "${FB_REPO}"

ok "Repositories ready:"
info "  TRM: ${TRM_DIR}"
info "  FB : ${FB_DIR}"
