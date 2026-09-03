#!/usr/bin/env bash
# Build adam-atan2 from the vendored source so its CUDA backend matches this
# machine's torch build and GPU architecture.
#
# Two build variants are available (see vendor/adam_atan2/setup.py):
#   patched (default) : C++20 + sm_80/86/89/90/100/120
#                       required for torch >= 2.9 headers and Blackwell GPUs
#                       such as this cluster's GB200 (sm_100).
#   legacy            : upstream 0.0.3 flags (C++17 + sm_80/86/89/90)
#                       for older torch builds or pre-Blackwell GPUs.
#
# Usage: build_adam_atan2.sh [fenchel_bregman|TinyRecursiveModels|both]
#   ADAM_ATAN2_BUILD=legacy bash setup/scripts/build_adam_atan2.sh   # legacy build
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
require_uv
load_cuda_toolkit

TARGET="${1:-both}"
BUILD_VARIANT="${ADAM_ATAN2_BUILD:-patched}"
case "${BUILD_VARIANT}" in
  patched|legacy) ;;
  *) die "Unknown ADAM_ATAN2_BUILD='${BUILD_VARIANT}'. Use 'patched' or 'legacy'" ;;
esac
VENDOR_SRC="${FB_DIR}/vendor/adam_atan2"
[[ -d "${VENDOR_SRC}" ]] || die "Vendored adam-atan2 source missing at ${VENDOR_SRC}"

install_into() {
  local project_dir="$1" name="$2"
  info "Building + installing adam-atan2 (${BUILD_VARIANT} build) into ${name} venv (this compiles CUDA kernels; may take a few minutes)"
  (cd "${project_dir}" && ADAM_ATAN2_BUILD="${BUILD_VARIANT}" uv pip install \
    --python .venv/bin/python \
    --no-cache --no-build-isolation --reinstall \
    "${VENDOR_SRC}")
  "${project_dir}/.venv/bin/python" -c "import torch, adam_atan2; print('adam_atan2 OK in ${name} venv')"
}

case "${TARGET}" in
  fenchel_bregman)      install_into "${FB_DIR}" "fenchel_bregman" ;;
  TinyRecursiveModels)  install_into "${TRM_DIR}" "TinyRecursiveModels" ;;
  both)
    install_into "${FB_DIR}" "fenchel_bregman"
    install_into "${TRM_DIR}" "TinyRecursiveModels"
    ;;
  *) die "Unknown target '${TARGET}'. Use fenchel_bregman | TinyRecursiveModels | both" ;;
esac

ok "adam-atan2 installed."
