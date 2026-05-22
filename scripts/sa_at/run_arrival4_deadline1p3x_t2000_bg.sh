#!/usr/bin/env bash
set -euo pipefail

# Dedicated background experiment:
#   SA-AT n300_w30_h200_t2000 only
#   methods: Greedy, HRLF, CAPAM, BASAR (ours)
#   arrival_rate = 4.0
#   deadline parameters = 1.3x current t2000 relaxed profile:
#     buffer 24 -> 31.2, horizon 75 -> 97.5
# Results are written to the same isolated arrival4/deadline1.3x root used by
# t1500, so the complementary table can consume both configs from one place.

PARALLELISM="${PARALLELISM:-16}"
THREADS_PER_JOB="${THREADS_PER_JOB:-2}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG="n300_w30_h200_t2000"
ARRIVAL_RATE="4.0"
DEADLINE_BUFFER="31.2"
DEADLINE_HORIZON="97.5"
URGENT_PROBABILITY="0.60"
PROFILE="arrival4_deadline1p3x"

RUN_ROOT="artifacts/results/sa_at_scaling_arrival4_deadline1p3x"
TMP_ROOT="${RUN_ROOT}/_env_parallel_outputs"
LOG_ROOT="${RUN_ROOT}/_env_parallel_logs"
JOB_FILE="${RUN_ROOT}/_env_parallel_jobs_t2000.tsv"

mkdir -p "$TMP_ROOT" "$LOG_ROOT"

echo "[$(date '+%F %T')] Building missing env-level jobs for ${CONFIG}..."
python - <<'PY' > "$JOB_FILE"
import json
from pathlib import Path

from scripts.sa_at.sa_at_scaling_config import get_dataset_dir, iter_scaling_configs

methods = ["greedy", "hrlf", "capam", "ours"]
config_name = "n300_w30_h200_t2000"
profile = "arrival4_deadline1p3x"
tmp_root = Path("artifacts/results/sa_at_scaling_arrival4_deadline1p3x/_env_parallel_outputs")

config = next(item for item in iter_scaling_configs() if item["name"] == config_name)


def has_valid_result(method, env_name):
    env_stem = Path(env_name).stem
    result_path = tmp_root / f"{method}_{config_name}_{env_stem}" / method / f"{config_name}_results.json"
    if not result_path.exists():
        return False
    try:
        rows = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(rows, list) or len(rows) != 1:
        return False
    row = rows[0]
    options = row.get("dynamic_task_options") or {}
    return (
        row.get("env_file") == env_name
        and row.get("config_name") == config_name
        and row.get("method") == method
        and abs(float(row.get("arrival_rate", -1.0)) - 4.0) < 1e-9
        and row.get("dynamic_task_profile") == profile
        and row.get("avg_deadline_violation_scope") == "completed_tasks"
        and abs(float(options.get("dynamic_deadline_buffer", -1.0)) - 31.2) < 1e-9
        and abs(float(options.get("dynamic_deadline_horizon", -1.0)) - 97.5) < 1e-9
    )


for env_file in sorted(get_dataset_dir(config).glob("env_*.pkl")):
    for method in methods:
        if not has_valid_result(method, env_file.name):
            print(method, config_name, env_file.name, sep="\t")
PY

JOB_COUNT="$(wc -l < "$JOB_FILE" | tr -d ' ')"
echo "[$(date '+%F %T')] Missing jobs: ${JOB_COUNT}; parallelism=${PARALLELISM}; threads/job=${THREADS_PER_JOB}"

if [[ "$JOB_COUNT" != "0" ]]; then
  xargs -a "$JOB_FILE" -n3 -P "$PARALLELISM" bash -c '
    set -euo pipefail
    METHOD="$1"
    CONFIG="$2"
    ENV_FILE="$3"
    ENV_STEM="${ENV_FILE%.pkl}"
    OUT_ROOT="artifacts/results/sa_at_scaling_arrival4_deadline1p3x/_env_parallel_outputs/${METHOD}_${CONFIG}_${ENV_STEM}"
    LOG="artifacts/results/sa_at_scaling_arrival4_deadline1p3x/_env_parallel_logs/${METHOD}_${CONFIG}_${ENV_STEM}.log"

    export CUDA_VISIBLE_DEVICES=""
    export OMP_NUM_THREADS="'"$THREADS_PER_JOB"'"
    export MKL_NUM_THREADS="'"$THREADS_PER_JOB"'"
    export OPENBLAS_NUM_THREADS="'"$THREADS_PER_JOB"'"
    export NUMEXPR_NUM_THREADS="'"$THREADS_PER_JOB"'"
    export TORCH_NUM_THREADS="'"$THREADS_PER_JOB"'"

    case "$METHOD" in
      greedy)
        SCRIPT="scripts/sa_at/evaluate_greedy_on_sa_at_dataset.py"
        EXTRA=()
        ;;
      hrlf)
        SCRIPT="scripts/sa_at/evaluate_hrlf_on_sa_at_dataset.py"
        EXTRA=(--folder-name save_baseline)
        ;;
      capam)
        SCRIPT="scripts/sa_at/evaluate_capam_on_sa_at_dataset.py"
        EXTRA=(--folder-name CAPAM_DYNAMIC)
        ;;
      ours)
        SCRIPT="scripts/sa_at/evaluate_my_model_on_sa_at_dataset.py"
        EXTRA=(--folder-name SAVE_5)
        ;;
      *)
        echo "Unknown method: $METHOD" >&2
        exit 2
        ;;
    esac

    conda run --no-capture-output -n hetero_mrta \
      python -u "$SCRIPT" "${EXTRA[@]}" \
      --config-name "$CONFIG" \
      --env-file "$ENV_FILE" \
      --output-root "$OUT_ROOT" \
      --arrival-rate "'"$ARRIVAL_RATE"'" \
      --dynamic-deadline-buffer "'"$DEADLINE_BUFFER"'" \
      --dynamic-deadline-horizon "'"$DEADLINE_HORIZON"'" \
      --dynamic-urgent-probability "'"$URGENT_PROBABILITY"'" \
      --dynamic-task-profile "'"$PROFILE"'" >"$LOG" 2>&1
  ' _
fi

echo "[$(date '+%F %T')] Merging env-level results for ${CONFIG}..."
python - <<'PY'
import json
import pickle
from pathlib import Path

methods = ["greedy", "hrlf", "capam", "ours"]
config_name = "n300_w30_h200_t2000"
profile = "arrival4_deadline1p3x"
tmp_root = Path("artifacts/results/sa_at_scaling_arrival4_deadline1p3x/_env_parallel_outputs")
result_root = Path("artifacts/results/sa_at_scaling_arrival4_deadline1p3x")


def env_index(row):
    try:
        return int(Path(row.get("env_file", "")).stem.split("_")[-1])
    except ValueError:
        return 10**9


for method in methods:
    rows = []
    for result_path in sorted(tmp_root.glob(f"{method}_{config_name}_env_*/*/{config_name}_results.json")):
        data = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(data, list) or len(data) != 1:
            raise RuntimeError(f"Unexpected result payload: {result_path}")
        row = data[0]
        if row.get("dynamic_task_profile") != profile:
            raise RuntimeError(f"Profile mismatch in {result_path}")
        if row.get("avg_deadline_violation_scope") != "completed_tasks":
            raise RuntimeError(f"ADV scope mismatch in {result_path}")
        rows.append(row)

    env_names = {row.get("env_file") for row in rows}
    if len(rows) != 50 or len(env_names) != 50:
        raise RuntimeError(f"{method} {config_name}: expected 50 unique rows, got {len(rows)} rows and {len(env_names)} envs")

    rows.sort(key=env_index)
    out_dir = result_root / method
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{config_name}_results.json"
    pkl_path = out_dir / f"{config_name}_results.pkl"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with pkl_path.open("wb") as f:
        pickle.dump(rows, f)

    n = len(rows)
    sr = sum(float(row.get("success_rate", 0.0)) for row in rows) / n * 100.0
    dsr = sum(float(row.get("deadline_satisfaction_rate", 0.0)) for row in rows) / n * 100.0
    adv = sum(float(row.get("avg_deadline_violation", 0.0)) for row in rows) / n
    ms = sum(float(row.get("makespan", 0.0)) for row in rows) / n
    pt = sum(float(row.get("total_planning_time", 0.0)) for row in rows) / n
    print(f"{method:6s} {config_name}: n={n} SR={sr:.1f}% DSR={dsr:.1f}% ADV={adv:.2f} MS={ms:.2f} PT={pt:.2f}s")
PY

echo "[$(date '+%F %T')] Regenerating complementary table..."
python tools/generate_sa_at_ma_at_complementary_table.py
if command -v latexmk >/dev/null 2>&1; then
  (
    cd docs/tables
    latexmk -g -pdf -interaction=nonstopmode sa_at_ma_at_complementary_preview.tex
    rm -f sa_at_ma_at_complementary_preview.aux \
      sa_at_ma_at_complementary_preview.fdb_latexmk \
      sa_at_ma_at_complementary_preview.fls \
      sa_at_ma_at_complementary_preview.log
  )
fi

echo "[$(date '+%F %T')] Done."
