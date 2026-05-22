#!/usr/bin/env bash
set -euo pipefail

# BASAR-only SA-AT dynamic intensity generalization experiment.
# Fixed benchmark config: n300_w30_h200_t1500.
# Variable: dynamic task arrival rate lambda.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-hetero_mrta}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data/testsets/sa_at_scaling}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/artifacts/results/sa_at_dynamic_intensity}"
CPU_ONLY="${CPU_ONLY:-1}"
CPU_THREADS="${CPU_THREADS:-32}"

CONFIG_NAME="${CONFIG_NAME:-n300_w30_h200_t1500}"
OURS_FOLDER="${OURS_FOLDER:-SAVE_5}"
ARRIVAL_RATES="${ARRIVAL_RATES:-0.5 1.0 2.0 3.0 4.0}"
SIMULATION_TIME_LIMIT="${SIMULATION_TIME_LIMIT:-}"
ENV_FILE="${ENV_FILE:-}"
RESUME="${RESUME:-1}"
GENERATE_TABLE="${GENERATE_TABLE:-1}"

lambda_key() {
  local rate="$1"
  local normalized
  normalized="$(printf "%.1f" "${rate}")"
  echo "${normalized//./p}"
}

has_checkpoint() {
  local folder="$1"
  [[ -f "${REPO_ROOT}/artifacts/models/${folder}/checkpoint.pth" || -f "${REPO_ROOT}/model/${folder}/checkpoint.pth" ]]
}

if ! has_checkpoint "${OURS_FOLDER}"; then
  echo "BASAR checkpoint not found for folder: ${OURS_FOLDER}" >&2
  echo "Expected artifacts/models/${OURS_FOLDER}/checkpoint.pth or model/${OURS_FOLDER}/checkpoint.pth" >&2
  exit 1
fi

export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS}"
export TORCH_NUM_THREADS="${CPU_THREADS}"
if [[ "${CPU_ONLY}" != "0" ]]; then
  export CUDA_VISIBLE_DEVICES=""
fi

echo "SA-AT dynamic intensity generalization"
echo "  repo root: ${REPO_ROOT}"
echo "  conda env: ${CONDA_ENV_NAME}"
echo "  dataset root: ${DATASET_ROOT}"
echo "  output root: ${OUTPUT_ROOT}"
echo "  config: ${CONFIG_NAME}"
echo "  model folder: ${OURS_FOLDER}"
echo "  arrival rates: ${ARRIVAL_RATES}"
if [[ "${CPU_ONLY}" != "0" ]]; then
  echo "  device mode: CPU-only"
else
  echo "  device mode: default torch device selection"
fi
echo "  CPU threads: ${CPU_THREADS}"
if [[ -n "${ENV_FILE}" ]]; then
  echo "  env file: ${ENV_FILE}"
fi
if [[ -n "${SIMULATION_TIME_LIMIT}" ]]; then
  echo "  simulation time limit: ${SIMULATION_TIME_LIMIT}"
fi
if [[ "${RESUME}" != "0" ]]; then
  echo "  resume mode: enabled"
else
  echo "  resume mode: disabled"
fi

read -r -a RATE_LIST <<< "${ARRIVAL_RATES}"
for rate in "${RATE_LIST[@]}"; do
  RATE_KEY="$(lambda_key "${rate}")"
  RATE_OUTPUT_ROOT="${OUTPUT_ROOT}/lambda_${RATE_KEY}"

  args=(
    "${REPO_ROOT}/scripts/sa_at/evaluate_my_model_on_sa_at_dataset.py"
    --folder-name "${OURS_FOLDER}"
    --dataset-root "${DATASET_ROOT}"
    --output-root "${RATE_OUTPUT_ROOT}"
    --config-name "${CONFIG_NAME}"
    --arrival-rate "${rate}"
  )

  if [[ -n "${SIMULATION_TIME_LIMIT}" ]]; then
    args+=(--simulation-time-limit "${SIMULATION_TIME_LIMIT}")
  fi
  if [[ -n "${ENV_FILE}" ]]; then
    args+=(--env-file "${ENV_FILE}")
  fi
  if [[ "${RESUME}" != "0" ]]; then
    args+=(--resume)
  fi

  echo
  echo "================================================================================"
  echo "Running BASAR at lambda=${rate}"
  echo "  output: ${RATE_OUTPUT_ROOT}"
  echo "================================================================================"
  conda run --no-capture-output -n "${CONDA_ENV_NAME}" python -u "${args[@]}"
done

if [[ "${GENERATE_TABLE}" != "0" ]]; then
  echo
  echo "================================================================================"
  echo "Generating dynamic intensity table"
  echo "================================================================================"
  conda run --no-capture-output -n "${CONDA_ENV_NAME}" python -u \
    "${REPO_ROOT}/tools/generate_sa_at_dynamic_intensity_table.py"
fi

echo
echo "SA-AT dynamic intensity generalization finished."
