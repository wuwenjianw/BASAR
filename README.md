# BASAR Dynamic MRTA

This repository contains the source code used for the RAL submission on dynamic heterogeneous multi-robot task allocation and scheduling. The code focuses on learning-based decision making for dynamic task arrivals, together with the evaluation scripts used for SA-BT, SA-AT, and MA-AT benchmark settings.

This package is prepared for anonymous code release. It intentionally avoids author, institution, and local-machine metadata.

Large generated datasets, pretrained checkpoints, intermediate figures, logs, and bulk result files are not included in this submission package because of the upload-size limit. They can be regenerated or placed back under the expected paths before running evaluation. Lightweight table sources under `docs/tables/` are included for reproducibility.

## Repository Structure

```text
.
├── env/                    # Dynamic MRTA environment and task simulation
├── scripts/
│   ├── sa_bt/              # Single-agent, bounded-time benchmark scripts
│   ├── sa_at/              # Single-agent, arrival-time scaling scripts
│   └── ma_at/              # Multi-agent, arrival-time benchmark scripts
├── tools/                  # Plotting, table, and GIF-generation utilities
├── docs/tables/            # Lightweight generated TeX/CSV table sources
├── artifacts/
│   ├── results/            # Generated evaluation JSON files, ignored by default
│   └── training/           # Lightweight training and ablation configs
├── attention.py            # Attention modules
├── model_capam.py          # CAPAM-style model components
├── decision_makers.py      # Policy and baseline decision interfaces
├── dynamic_driver.py       # Main dynamic training entry point
├── reward_ablation.py      # Reward-ablation training logic
└── module_ablation.py      # Module-ablation training logic
```

## Requirements

The code was organized for Python 3.9+ and PyTorch-based experiments. A Python 3.10 environment is recommended.

Main packages:

- `python >= 3.9`
- `torch >= 1.13`
- `numpy >= 1.23`
- `scipy >= 1.9`
- `pandas >= 1.5`
- `ray >= 2.0`
- `matplotlib >= 3.6`

Optional packages:

- `ortools >= 9.0` for the CTAS-D route-planning baseline.
- `torch-sparse` for CAPAM-related components. Install a build that matches the local PyTorch and CUDA versions.

Example setup:

```bash
conda create -n basar python=3.10
conda activate basar
pip install torch numpy scipy pandas ray matplotlib ortools
```

If CAPAM experiments are needed, install `torch-sparse` separately following the PyTorch Geometric installation instructions for the target CUDA/PyTorch version.

## Data and Checkpoints

Generated benchmark datasets are expected under:

```text
data/testsets/sa_bt/
data/testsets/sa_at_scaling/
data/testsets/ma_at_dynamic/
```

Model checkpoints are expected under:

```text
artifacts/models/<MODEL_NAME>/checkpoint.pth
```

These files are intentionally excluded from this lightweight submission package. Evaluation commands that load a trained model require the corresponding checkpoint to be restored first.

## Common Commands

Generate benchmark datasets:

```bash
python -u scripts/sa_bt/generate_sa_bt_dataset.py
python -u scripts/sa_at/generate_sa_at_scaling_dataset.py
python -u scripts/ma_at/generate_ma_at_dynamic_dataset.py
```

Train the main dynamic model:

```bash
python -u dynamic_driver.py
```

Run ablation training:

```bash
python -u run_reward_ablation.py --preset no_shared --folder-name ABL_NO_SHARED
python -u run_module_ablation.py --preset global_mlp --folder-name MODABL_GLOBAL_MLP --model-name myself
```

Evaluate models and baselines:

```bash
python -u scripts/sa_bt/evaluate_my_model_dynamic.py
python -u scripts/sa_bt/evaluate_ctasd_dynamic.py
python -u scripts/sa_bt/evaluate_taco_dynamic.py
python -u scripts/sa_bt/evaluate_greedy_on_sa_bt_dataset.py

python -u scripts/sa_at/evaluate_my_model_on_sa_at_dataset.py
python -u scripts/ma_at/evaluate_my_model_on_ma_at_dataset.py
```

Scenario-level batch evaluation helpers are also available:

```bash
bash scripts/sa_at/run_all_evals.sh
bash scripts/ma_at/run_all_evals.sh
```

Run the BASAR-only SA-AT dynamic-intensity generalization experiment:

```bash
CPU_ONLY=1 CPU_THREADS=32 bash scripts/sa_at/run_dynamic_intensity_generalization.sh
```

Regenerate compact table sources after evaluation results are available:

```bash
python -u tools/sa_bt/generate_ieee_tables.py
python -u tools/generate_sa_at_ma_at_complementary_table.py
python -u tools/generate_sa_at_dynamic_intensity_table.py
```

## Outputs

Evaluation scripts write JSON result files to `artifacts/results/` by default. Training and ablation metadata are stored under `artifacts/training/`. Plot, table, and GIF utilities under `tools/` can regenerate paper-facing figures, tables, and visualizations after the required datasets, checkpoints, and result files are available.

## Notes

- Run commands from the repository root so that relative paths resolve correctly.
- CUDA is optional but recommended for model training.
- The repository has been cleaned for anonymous code submission: backup directories, historical checkpoints, generated datasets, bulk results, logs, and intermediate media artifacts were removed.
