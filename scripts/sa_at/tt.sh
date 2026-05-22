conda run -n hetero_mrta python scripts/sa_at/generate_sa_at_scaling_dataset.py

ARRIVAL_RATE=2.0 CONDA_ENV_NAME=hetero_mrta bash scripts/sa_at/run_all_evals.sh
