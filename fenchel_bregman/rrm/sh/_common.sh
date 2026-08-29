#!/usr/bin/env bash

RRM_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
RRM_PYTHON_BIN="${PYTHON_BIN:-python}"
RRM_DEVICE="${DEVICE:-cuda}"
export PYTHONPATH="${RRM_PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

run_table1_evaluation() {
  local model="$1"
  local task="$2"
  local checkpoint="$3"
  local dataset="$4"
  local output="$5"
  local candidates="$6"
  local depth="$7"
  local batch_size="$8"
  local shards="$9"
  shift 9
  local -a extra_args=("$@")

  [[ -f "${checkpoint}" ]] || { echo "Missing checkpoint: ${checkpoint}" >&2; return 2; }
  [[ -d "${dataset}" ]] || { echo "Missing dataset: ${dataset}" >&2; return 2; }

  if (( shards == 1 )); then
    "${RRM_PYTHON_BIN}" -m rrm.evaluation \
      --model "${model}" --task "${task}" --preset paper \
      --checkpoint "${checkpoint}" --dataset "${dataset}" --output "${output}" \
      --candidate-count "${candidates}" --depth "${depth}" \
      --batch-size "${batch_size}" --seed 0 --device "${RRM_DEVICE}" \
      "${extra_args[@]}"
    return
  fi

  local shard shard_output
  local -a outcomes=()
  mkdir -p "${output}/shards"
  for ((shard = 0; shard < shards; shard++)); do
    shard_output="${output}/shards/${shard}"
    "${RRM_PYTHON_BIN}" -m rrm.evaluation \
      --model "${model}" --task "${task}" --preset paper \
      --checkpoint "${checkpoint}" --dataset "${dataset}" --output "${shard_output}" \
      --candidate-count "${candidates}" --depth "${depth}" \
      --batch-size "${batch_size}" --num-shards "${shards}" --shard-index "${shard}" \
      --seed 0 --device "${RRM_DEVICE}" "${extra_args[@]}"
    outcomes+=("${shard_output}/outcomes.pt")
  done

  "${RRM_PYTHON_BIN}" -m rrm.evaluation \
    --model "${model}" --task "${task}" --preset paper \
    --aggregate-only "${outcomes[@]}" --output "${output}/aggregate" \
    --candidate-count "${candidates}" "${extra_args[@]}"
}
