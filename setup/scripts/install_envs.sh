#!/usr/bin/env bash
# Create the Python virtual environments for both codebases and install deps.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
require_uv

# --- TinyRecursiveModels ---------------------------------------------------
if [[ ! -x "${TRM_PYTHON}" ]]; then
  info "Creating TRM venv (python 3.12)"
  (cd "${TRM_DIR}" && uv venv --python 3.12 .venv)
else
  ok "TRM venv already exists"
fi

info "Installing TRM dependencies (uv sync)"
(cd "${TRM_DIR}" && uv sync --active 2>/dev/null || uv sync)

# --- fenchel_bregman -------------------------------------------------------
if [[ ! -x "${FB_PYTHON}" ]]; then
  info "Creating fenchel_bregman venv (python 3.12)"
  (cd "${FB_DIR}" && uv venv --python 3.12 .venv)
else
  ok "fenchel_bregman venv already exists"
fi

info "Installing fenchel_bregman dependencies (uv sync)"
(cd "${FB_DIR}" && uv sync --active 2>/dev/null || uv sync)

ok "Environments ready:"
info "  TRM: ${TRM_PYTHON}"
info "  FB : ${FB_PYTHON}"
