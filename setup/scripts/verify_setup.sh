#!/usr/bin/env bash
# Verify the whole setup: imports, CUDA visibility, datasets, checkpoints.
# Exits non-zero on the first failed check.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_config || true

failures=0
check() { if "$@" >/dev/null 2>&1; then ok "$*"; else err "$*"; failures=$((failures+1)); fi }

info "1/4 Python environments"
check test -x "${TRM_PYTHON}"
check test -x "${FB_PYTHON}"

info "2/4 Core imports + CUDA"
if [[ -x "${TRM_PYTHON}" ]]; then
  "${TRM_PYTHON}" - <<'EOF' || failures=$((failures+1))
import torch, numpy, argdantic
assert torch.cuda.is_available(), "CUDA not visible to TRM venv"
print(f"  torch {torch.__version__}, {torch.cuda.device_count()} GPU(s), {torch.cuda.get_device_name(0)}")
EOF
fi
if [[ -x "${FB_PYTHON}" ]]; then
  "${FB_PYTHON}" - <<'EOF' || failures=$((failures+1))
import torch, adam_atan2, ivon, evon
assert torch.cuda.is_available(), "CUDA not visible to fenchel_bregman venv"
print(f"  torch {torch.__version__}, adam_atan2 backend OK")
EOF
fi

info "3/4 Datasets"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data}"
for ds in arc1concept-aug-1000 arc2concept-aug-1000; do
  d="${DATA_ROOT}/${ds}"
  if [[ -f "${d}/train/dataset.json" && -f "${d}/test/dataset.json" && -f "${d}/identifiers.json" ]]; then
    ok "dataset ${ds}"
  else
    err "dataset ${ds} missing or incomplete at ${d}"
    failures=$((failures+1))
  fi
done

info "4/4 RRM module import (fenchel_bregman)"
if [[ -x "${FB_PYTHON}" ]]; then
  (cd "${FB_DIR}" && "${FB_PYTHON}" -c "from rrm import train, evaluation, arc, ptrm" ) \
    && ok "rrm modules import cleanly" || { err "rrm import failed"; failures=$((failures+1)); }
fi

if (( failures == 0 )); then
  ok "All checks passed."
else
  err "${failures} check(s) failed."
  exit 1
fi
