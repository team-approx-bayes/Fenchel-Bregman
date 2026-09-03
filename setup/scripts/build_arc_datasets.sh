#!/usr/bin/env bash
# Build the ARC-AGI-1 and ARC-AGI-2 datasets from the raw Kaggle JSON files
# that ship inside TinyRecursiveModels/kaggle/combined/.
#
# Usage: build_arc_datasets.sh [DATA_ROOT]
#   DATA_ROOT defaults to <repo>/data (or the DATA_ROOT saved by setup.sh).
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
require_trm_python

DATA_ROOT="${1:-${DATA_ROOT:-${ROOT}/data}}"
mkdir -p "${DATA_ROOT}"

# ARC-AGI-1: training + evaluation + concept subsets, test set = evaluation
info "Building ARC-AGI-1 -> ${DATA_ROOT}/arc1concept-aug-1000"
(cd "${TRM_DIR}" && "${TRM_PYTHON}" -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir "${DATA_ROOT}/arc1concept-aug-1000" \
  --subsets training evaluation concept \
  --test-set-name evaluation)

# ARC-AGI-2: training2 + evaluation2 + concept subsets, test set = evaluation2
# NOTE: ARC-AGI-2 training data contains some ARC-AGI-1 eval data, so do not
# train on both and evaluate both (see TinyRecursiveModels README).
info "Building ARC-AGI-2 -> ${DATA_ROOT}/arc2concept-aug-1000"
(cd "${TRM_DIR}" && "${TRM_PYTHON}" -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir "${DATA_ROOT}/arc2concept-aug-1000" \
  --subsets training2 evaluation2 concept \
  --test-set-name evaluation2)

ok "ARC datasets built under ${DATA_ROOT}:"
du -sh "${DATA_ROOT}"/arc1concept-aug-1000 "${DATA_ROOT}"/arc2concept-aug-1000 2>/dev/null || true
