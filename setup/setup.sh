#!/usr/bin/env bash
# Interactive setup for the posterior-decision-reasoning project.
#
# Guides you through:
#   1. Where to store datasets (saved to setup/setup.env and reused)
#   2. Which components to install (code, envs, datasets, adam-atan2)
#   3. Verification of the final state
#
# Every step is idempotent and can also be run standalone via setup/scripts/*.
set -euo pipefail

SETUP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="${SETUP_DIR}/scripts"
# shellcheck disable=SC1091
source "${SCRIPTS}/common.sh"

# ---------------------------------------------------------------------------
# Load previous configuration, if any
# ---------------------------------------------------------------------------
if load_config; then
  ok "Found existing configuration: ${CONFIG_FILE}"
  info "  DATA_ROOT = ${DATA_ROOT:-<unset>}"
  info "  CUDA_MODULE = ${CUDA_MODULE:-<cluster default>}"
else
  info "No previous configuration found (${CONFIG_FILE})."
fi

echo
echo "${C_BOLD}What would you like to do?${C_RESET}"
echo "  1) Full setup  (configure + code + environments + datasets + adam-atan2 + verify)"
echo "  2) Configure only  (choose dataset location and CUDA module)"
echo "  3) Code only       (clone/update TinyRecursiveModels and fenchel_bregman)"
echo "  4) Environments    (create venvs and install dependencies)"
echo "  5) Datasets        (build ARC-AGI-1/2 and/or Sudoku/Maze into DATA_ROOT)"
echo "  6) adam-atan2      (build CUDA optimizer for this GPU architecture)"
echo "  7) Verify          (check envs, CUDA, datasets, imports)"
echo "  8) TRM checkpoints (download official ARC-AGI-1/2 TRM weights into models/)"
echo "  q) Quit"
echo
read -r -p "Choice [1]: " choice </dev/tty
choice="${choice:-1}"

run_step() {
  local script="$1"; shift
  echo
  info "Running ${script} $*"
  bash "${SCRIPTS}/${script}" "$@"
}

# ---------------------------------------------------------------------------
# Step: configuration (always first for choices 1 and 2)
# ---------------------------------------------------------------------------
configure() {
  echo
  info "Configuration"
  echo
  echo "${C_BOLD}Dataset storage location${C_RESET}"
  echo "  Datasets are large (ARC-AGI-1 ~7 GB, ARC-AGI-2 ~9 GB, Sudoku ~1 GB, Maze ~1 GB)."
  echo "  Pick a path with enough free space, ideally on a fast/scratch filesystem."
  ask_path "  Where should datasets be stored" "${DATA_ROOT:-${ROOT}/data}"
  DATA_ROOT="${REPLY}"
  mkdir -p "${DATA_ROOT}"

  echo
  echo "${C_BOLD}CUDA toolkit${C_RESET}"
  echo "  Needed to compile adam-atan2. On clusters with environment modules,"
  echo "  give the module name (e.g. cuda/13.2.2). Leave empty to skip module load."
  read -r -p "  CUDA module name [${CUDA_MODULE:-cuda/13.2.2}]: " reply </dev/tty
  CUDA_MODULE="${reply:-${CUDA_MODULE:-cuda/13.2.2}}"

  save_config "DATA_ROOT" "${DATA_ROOT}" "CUDA_MODULE" "${CUDA_MODULE}"
  export DATA_ROOT CUDA_MODULE
  ok "Saved configuration to ${CONFIG_FILE}"
}

# ---------------------------------------------------------------------------
# Step: dataset selection (choice 5)
# ---------------------------------------------------------------------------
choose_datasets() {
  echo
  echo "${C_BOLD}Which datasets should be built into ${DATA_ROOT}?${C_RESET}"
  echo "  1) ARC-AGI-1 + ARC-AGI-2   (needed for rrm/sh/agi-1 and rrm/sh/agi-2)"
  echo "  2) Sudoku-Extreme + Maze-Hard  (Table 1 tasks; downloads from Hugging Face)"
  echo "  3) All of the above"
  read -r -p "Choice [3]: " dchoice </dev/tty
  dchoice="${dchoice:-3}"
  case "${dchoice}" in
    1) run_step "build_arc_datasets.sh" "${DATA_ROOT}" ;;
    2) run_step "build_table1_datasets.sh" "${DATA_ROOT}" ;;
    3) run_step "build_arc_datasets.sh" "${DATA_ROOT}"
       run_step "build_table1_datasets.sh" "${DATA_ROOT}" ;;
    *) warn "No datasets selected." ;;
  esac
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
case "${choice}" in
  1)
    configure
    run_step "clone_repos.sh"
    run_step "install_envs.sh"
    choose_datasets
    run_step "build_adam_atan2.sh" both
    run_step "download_trm_checkpoints.sh"
    run_step "verify_setup.sh"
    echo
    ok "Full setup complete. See README.md for training commands."
    ;;
  2) configure ;;
  3) run_step "clone_repos.sh" ;;
  4) run_step "install_envs.sh" ;;
  5)
    load_config || warn "No configuration yet; using defaults."
    DATA_ROOT="${DATA_ROOT:-${ROOT}/data}"
    choose_datasets
    ;;
  6)
    load_config || true
    echo
    echo "${C_BOLD}Build adam-atan2 into which environment?${C_RESET}"
    echo "  1) fenchel_bregman only"
    echo "  2) TinyRecursiveModels only"
    echo "  3) Both"
    read -r -p "Choice [3]: " achoice </dev/tty
    achoice="${achoice:-3}"
    echo
    echo "${C_BOLD}Build variant${C_RESET}"
    echo "  patched (default) : C++20 + sm_100/120 — required for torch >= 2.9 and Blackwell GPUs (GB200)"
    echo "  legacy            : upstream flags (C++17 + sm_80/86/89/90) — older torch / pre-Blackwell GPUs"
    read -r -p "Variant [patched]: " variant </dev/tty
    variant="${variant:-patched}"
    case "${variant}" in
      legacy) export ADAM_ATAN2_BUILD=legacy ;;
      patched) unset ADAM_ATAN2_BUILD ;;
      *) warn "Unknown variant '${variant}', using patched."; unset ADAM_ATAN2_BUILD ;;
    esac
    case "${achoice}" in
      1) run_step "build_adam_atan2.sh" fenchel_bregman ;;
      2) run_step "build_adam_atan2.sh" TinyRecursiveModels ;;
      *) run_step "build_adam_atan2.sh" both ;;
    esac
    ;;
  7) run_step "verify_setup.sh" ;;
  8) run_step "download_trm_checkpoints.sh" ;;
  q|Q) info "Bye." ;;
  *) die "Unknown choice '${choice}'" ;;
esac
