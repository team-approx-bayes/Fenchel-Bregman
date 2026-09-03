#!/usr/bin/env bash
# Shared helpers for the setup scripts. Source this file, never execute it.
#
# Every script in setup/scripts/ (and setup/setup.sh) sources this file to get
# consistent logging, prompting, and configuration persistence.

# Repo root: two levels up from this file (setup/scripts/common.sh).
: "${ROOT:=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
export ROOT

CONFIG_FILE="${ROOT}/setup/setup.env"

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
if [[ -t 1 ]]; then
  C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'; C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_RED=$'\033[31m'; C_BLUE=$'\033[34m'
else
  C_RESET=""; C_BOLD=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_BLUE=""
fi

info()  { printf '%s[setup]%s %s\n' "${C_BLUE}" "${C_RESET}" "$*"; }
ok()    { printf '%s[ ok ]%s %s\n' "${C_GREEN}" "${C_RESET}" "$*"; }
warn()  { printf '%s[warn]%s %s\n' "${C_YELLOW}" "${C_RESET}" "$*"; }
err()   { printf '%s[fail]%s %s\n' "${C_RED}" "${C_RESET}" "$*" >&2; }
die()   { err "$*"; exit 1; }

# ---------------------------------------------------------------------------
# Configuration persistence (setup/setup.env, KEY=VALUE lines)
# ---------------------------------------------------------------------------
load_config() {
  if [[ -f "${CONFIG_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${CONFIG_FILE}"
    return 0
  fi
  return 1
}

save_config() {
  # save_config KEY VALUE [KEY VALUE ...]
  touch "${CONFIG_FILE}"
  local -A kv=()
  while [[ $# -ge 2 ]]; do kv["$1"]="$2"; shift 2; done
  local key
  for key in "${!kv[@]}"; do
    if grep -q "^${key}=" "${CONFIG_FILE}" 2>/dev/null; then
      sed -i "s|^${key}=.*|${key}=${kv[$key]}|" "${CONFIG_FILE}"
    else
      printf '%s=%s\n' "${key}" "${kv[$key]}" >> "${CONFIG_FILE}"
    fi
  done
}

# ---------------------------------------------------------------------------
# Interactive prompts (only used by setup.sh)
# ---------------------------------------------------------------------------
ask_path() {
  # ask_path PROMPT DEFAULT_VAR_NAME  -> sets REPLY
  local prompt="$1" default="$2"
  read -r -e -p "${prompt} [${default}]: " REPLY </dev/tty || REPLY="${default}"
  REPLY="${REPLY:-${default}}"
  # Expand leading ~
  REPLY="${REPLY/#\~/$HOME}"
}

ask_choice() {
  # ask_choice PROMPT  -> sets REPLY
  read -r -e -p "${prompt:-$1}" REPLY </dev/tty || REPLY=""
}

confirm() {
  # confirm PROMPT [default=y|n] -> return 0 on yes
  local prompt="$1" default="${2:-y}" answer
  if [[ "${default}" == "y" ]]; then
    read -r -p "${prompt} [Y/n]: " answer </dev/tty || return 1
    [[ -z "${answer}" || "${answer,,}" == "y" || "${answer,,}" == "yes" ]]
  else
    read -r -p "${prompt} [y/N]: " answer </dev/tty || return 1
    [[ "${answer,,}" == "y" || "${answer,,}" == "yes" ]]
  fi
}

# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------
TRM_DIR="${ROOT}/TinyRecursiveModels"
FB_DIR="${ROOT}/fenchel_bregman"
TRM_PYTHON="${TRM_DIR}/.venv/bin/python"
FB_PYTHON="${FB_DIR}/.venv/bin/python"

require_uv() {
  command -v uv >/dev/null 2>&1 || die "uv is required but not on PATH. Install it from https://docs.astral.sh/uv/ first."
}

require_trm() {
  [[ -d "${TRM_DIR}" ]] || die "TinyRecursiveModels not found at ${TRM_DIR}. Run the 'Code' setup step (or setup/scripts/clone_repos.sh) first."
}

require_trm_python() {
  require_trm
  [[ -x "${TRM_PYTHON}" ]] || die "TRM venv missing at ${TRM_DIR}/.venv. Run the 'Environments' setup step (or setup/scripts/install_envs.sh) first."
}

# Load the CUDA toolkit module on clusters that use environment modules.
# Sets CUDA_HOME as a side effect. Safe to call when CUDA is already set up.
load_cuda_toolkit() {
  if command -v nvcc >/dev/null 2>&1 && [[ -n "${CUDA_HOME:-}" ]]; then
    info "Using existing CUDA toolkit: ${CUDA_HOME}"
    return 0
  fi
  if command -v module >/dev/null 2>&1; then
    local cuda_module="${CUDA_MODULE:-cuda/13.2.2}"
    info "Loading CUDA module '${cuda_module}'"
    module load "${cuda_module}" || warn "module load ${cuda_module} failed; set CUDA_MODULE or load a CUDA module manually"
  fi
  if command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME="${CUDA_HOME:-$(dirname -- "$(dirname -- "$(command -v nvcc)")")}"
    info "CUDA_HOME=${CUDA_HOME} (nvcc $(nvcc --version | tail -1 | grep -o 'release [0-9.]*'))"
  else
    die "nvcc not found. Load a CUDA module (e.g. 'module load cuda/13.2.2') or install the CUDA toolkit."
  fi
}
