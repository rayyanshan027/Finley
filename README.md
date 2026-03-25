# Finley

Starter ML repo for working with CRCNS HC-6 data already extracted on Eureka HPC.

Project narrative and ongoing status live in [PROJECT_NARRATIVE.md](/Users/rayyanshan/IdeaProjects/Finley/PROJECT_NARRATIVE.md).

## What this gives you

- A clean Python package under `src/finley`
- Config-driven access to the HC-6 extracted dataset root
- A dataset inventory script that recognizes HC-6 file families like `spikes`, `pos`, `rawpos`, `task`, `cellinfo`, `tetinfo`, and `EEG`
- A minimal baseline training script that consumes the inventory output

## Expected dataset root

Point the project at the extracted HC-6 directory, for example:

```bash
/projects/clg_24609687d5dd/hc6/raw/extracted
```

If you only want to start with the first animal you mentioned, you can also point it at:

```bash
/projects/clg_24609687d5dd/hc6/raw/extracted/Bon
```

## Quick start

Create or reuse a Python 3.10+ environment, then install the package in editable mode:

```bash
python -m pip install -e ".[dev]"
```

Copy the example config if you want a machine-specific version:

```bash
cp configs/hc6.local.example.json configs/hc6.local.json
```

Build an inventory of the dataset:

```bash
PYTHONPATH=src python scripts/build_inventory.py --config configs/hc6.local.json
```

Run the starter baseline:

```bash
PYTHONPATH=src python scripts/train_baseline.py --config configs/hc6.local.json
```

Inspect one real HC-6 session before writing model code:

```bash
PYTHONPATH=src python scripts/inspect_mat.py --config configs/hc6.local.json --animal Bon --session 3
```

Load one session as a smoke test once `scipy` is available:

```bash
PYTHONPATH=src python scripts/load_session.py --config configs/hc6.local.json --animal Bon --session 3
```

Generate a parsed per-epoch summary you can actually model from:

```bash
PYTHONPATH=src python scripts/summarize_session.py --config configs/hc6.local.json --animal Bon --session 3
```

Export flattened CSV tables for one session:

```bash
PYTHONPATH=src python scripts/export_session_tables.py --config configs/hc6.local.json --animal Bon --session 3
```

Export flattened CSV tables across all discovered sessions for one animal:

```bash
PYTHONPATH=src python scripts/export_session_tables.py --config configs/hc6.local.json --animal Bon --all-sessions
```

Build a run-only modeling table across all discovered sessions:

```bash
PYTHONPATH=src python scripts/build_model_table.py --config configs/hc6.local.json --animal Bon
```

The run-cell model table now includes position-derived epoch features such as duration, mean/std/max speed, moving fraction, and spatial range, plus firing-rate targets when epoch duration is available.

Train the first run-cell baseline with held-out-session evaluation:

```bash
PYTHONPATH=src python scripts/train_run_cell_baseline.py --input data/processed/bon_run_cell_model_table.csv
```

Recommended default benchmark:

```bash
PYTHONPATH=src python scripts/train_run_cell_baseline.py --input data/processed/bon_run_cell_model_table.csv
```

The default run-cell baseline now uses `movement_summaries`, `log_firing_rate_hz` as the target, and ridge alpha `100`.
The baseline trainer standardizes numeric features, fits a ridge-regularized linear model, and filters rows with missing target values when the chosen target depends on epoch duration.

## Project layout

```text
configs/               JSON configs
data/                  generated inventories and processed artifacts
notebooks/             ad hoc exploration
scripts/               runnable entrypoints
src/finley/data/       dataset config, scanning, and inventory logic
src/finley/models/     starter baseline model code
tests/                 unit tests
```

## Notes

- The inventory output now includes `modality`, `session`, and `top_level_dir` columns derived from filenames like `bonspikes03.mat`.
- `scripts/inspect_mat.py` is the intended first step for understanding the actual `.mat` contents on Eureka.
- `scripts/summarize_session.py` unwraps the HC-6 nesting and reports epoch-level task and spike counts.
- `scripts/export_session_tables.py` writes an epoch table and a cell-level spike table for one session.
- `scripts/export_session_tables.py --all-sessions` writes combined tables across all discovered sessions for one animal.
- `scripts/build_model_table.py` builds a run-only cell-level modeling table across all discovered sessions.
- `scripts/train_run_cell_baseline.py` trains a simple held-out-session regression baseline on the run-cell model table, with `log_firing_rate_hz` as the default target and `movement_summaries` as the default feature set.
- Duration-dependent targets such as `firing_rate_hz` and `log_firing_rate_hz` may be missing for some exported rows; the trainer filters those rows before fitting and reports the kept counts in its metrics output.
- Current project benchmark: default run-cell baseline on held-out session `10` with MAE `0.3731`, RMSE `0.6103`.
- Reading `.mat` files requires `scipy`, which is not vendored into this repo.
