#!/usr/bin/env bash
# ARRIVAL_RATE=2.0 CONDA_ENV_NAME=hetero_mrta bash scripts/ma_at/run_all_evals.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-hetero_mrta}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data/testsets/ma_at_dynamic}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/artifacts/results/ma_at_dynamic}"

OURS_FOLDER="${OURS_FOLDER:-SAVE_5}"
HRLF_FOLDER="${HRLF_FOLDER:-save_baseline}"
CAPAM_FOLDER="${CAPAM_FOLDER:-CAPAM_DYNAMIC}"

ARRIVAL_RATE="${ARRIVAL_RATE:-}"
SIMULATION_TIME_LIMIT="${SIMULATION_TIME_LIMIT:-}"
RESUME="${RESUME:-1}"

COMMON_ARGS=(--dataset-root "${DATASET_ROOT}" --output-root "${OUTPUT_ROOT}")
if [[ -n "${ARRIVAL_RATE}" ]]; then
  COMMON_ARGS+=(--arrival-rate "${ARRIVAL_RATE}")
fi
if [[ -n "${SIMULATION_TIME_LIMIT}" ]]; then
  COMMON_ARGS+=(--simulation-time-limit "${SIMULATION_TIME_LIMIT}")
fi
if [[ "${RESUME}" != "0" ]]; then
  COMMON_ARGS+=(--resume)
fi

has_checkpoint() {
  local folder="$1"
  [[ -f "${REPO_ROOT}/artifacts/models/${folder}/checkpoint.pth" ]]
}

run_eval() {
  local label="$1"
  shift
  echo
  echo "================================================================================"
  echo "Running ${label}"
  echo "================================================================================"
  conda run --no-capture-output -n "${CONDA_ENV_NAME}" python "$@"
}

echo "MA-AT dynamic one-click evaluation"
echo "  repo root: ${REPO_ROOT}"
echo "  conda env: ${CONDA_ENV_NAME}"
echo "  dataset root: ${DATASET_ROOT}"
echo "  output root: ${OUTPUT_ROOT}"
if [[ -n "${ARRIVAL_RATE}" ]]; then
  echo "  override arrival rate: ${ARRIVAL_RATE}"
fi
if [[ -n "${SIMULATION_TIME_LIMIT}" ]]; then
  echo "  override simulation time limit: ${SIMULATION_TIME_LIMIT}"
fi
if [[ "${RESUME}" != "0" ]]; then
  echo "  resume mode: enabled"
else
  echo "  resume mode: disabled"
fi

run_eval "Greedy" \
  "${REPO_ROOT}/scripts/ma_at/evaluate_greedy_on_ma_at_dataset.py" \
  "${COMMON_ARGS[@]}"

if has_checkpoint "${HRLF_FOLDER}"; then
  run_eval "HRLF" \
    "${REPO_ROOT}/scripts/ma_at/evaluate_hrlf_on_ma_at_dataset.py" \
    --folder-name "${HRLF_FOLDER}" \
    "${COMMON_ARGS[@]}"
else
  echo "Skipping HRLF: checkpoint not found for ${HRLF_FOLDER}"
fi

if has_checkpoint "${OURS_FOLDER}"; then
  run_eval "Ours" \
    "${REPO_ROOT}/scripts/ma_at/evaluate_my_model_on_ma_at_dataset.py" \
    --folder-name "${OURS_FOLDER}" \
    "${COMMON_ARGS[@]}"
else
  echo "Skipping Ours: checkpoint not found for ${OURS_FOLDER}"
fi

if has_checkpoint "${CAPAM_FOLDER}"; then
  run_eval "CAPAM" \
    "${REPO_ROOT}/scripts/ma_at/evaluate_capam_on_ma_at_dataset.py" \
    --folder-name "${CAPAM_FOLDER}" \
    "${COMMON_ARGS[@]}"
else
  echo "Skipping CAPAM: checkpoint not found for ${CAPAM_FOLDER}"
fi

echo
echo "All requested MA-AT evaluations finished."
