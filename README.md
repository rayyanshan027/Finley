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

Train the pure-Python nonlinear baseline with held-out-session evaluation:

```bash
PYTHONPATH=src python scripts/train_run_cell_nonlinear.py --input data/processed/bon_run_cell_model_table.csv
```

Run leave-one-session-out evaluation for the nonlinear baseline:

```bash
PYTHONPATH=src python scripts/train_run_cell_nonlinear.py --input data/processed/bon_run_cell_model_table.csv --leave-one-session-out
```

Run the current best nonlinear benchmark candidate:

```bash
PYTHONPATH=src python scripts/train_run_cell_nonlinear.py \
  --input data/processed/bon_run_cell_model_table.csv \
  --leave-one-session-out \
  --target log_firing_rate_hz \
  --feature-groups movement_summaries population_context cell_metadata \
  --output artifacts/run_cell_nonlinear_movement_population_cell.json
```

Inspect hard sessions against easier comparison sessions using quantiles and outlier rows:

```bash
PYTHONPATH=src python scripts/inspect_hard_sessions.py \
  --input data/processed/bon_run_cell_model_table.csv \
  --hard-sessions 6 7 9 \
  --easy-sessions 3 4 8
```

Inspect leave-one-session-out residuals for the current best nonlinear candidate:

```bash
PYTHONPATH=src python scripts/inspect_hard_session_residuals_nonlinear.py \
  --input data/processed/bon_run_cell_model_table.csv \
  --output artifacts/hard_session_residuals_nonlinear.json \
  --sessions 6 7 9 \
  --target log_firing_rate_hz \
  --feature-groups movement_summaries population_context cell_metadata
```

Run the session-adaptation benchmark using 0, 1, or 2 labeled epochs from the held-out session:

```bash
PYTHONPATH=src python scripts/run_session_adaptation_experiment.py \
  --input data/processed/bon_run_cell_model_table.csv \
  --output artifacts/session_adaptation_experiment.json \
  --sessions 6 7 9 \
  --adaptation-epochs 0 1 2 \
  --target log_firing_rate_hz \
  --feature-groups movement_summaries population_context cell_metadata
```

Recommended default benchmark:

```bash
PYTHONPATH=src python scripts/train_run_cell_nonlinear.py \
  --input data/processed/bon_run_cell_model_table.csv \
  --leave-one-session-out \
  --target log_firing_rate_hz \
  --feature-groups movement_summaries population_context cell_metadata
```

Current recommended setting:

- model: pure-Python nonlinear forest-style baseline
- target: `log_firing_rate_hz`
- feature groups: `movement_summaries`, `population_context`, `cell_metadata`
- LOSO mean MAE: `0.4265`
- LOSO mean RMSE: `0.5764`

Current session-adaptive result:

- adding 1 labeled epoch from the held-out session materially improves sessions `6`, `7`, and `9`
- adding 2 labeled epochs improves them further, indicating the remaining gap is substantially session-specific

Simple linear reference:

- model: ridge baseline
- target: `log_firing_rate_hz`
- feature groups: `movement_summaries`
- held-out session `10`: MAE `0.3731`, RMSE `0.6103`

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
- `scripts/train_run_cell_nonlinear.py` trains a pure-Python random-forest-style nonlinear baseline on the same run-cell model table and supports leave-one-session-out evaluation.
- `scripts/inspect_hard_sessions.py` compares hard and easy sessions using quantiles and writes top outlier cell/epoch rows for targeted inspection.
- `scripts/inspect_hard_session_residuals.py` inspects held-out-session residuals so repeated hard-case cells can be identified directly.
- `scripts/inspect_hard_session_residuals_nonlinear.py` inspects held-out-session residuals for the nonlinear benchmark candidate.
- `scripts/run_session_adaptation_experiment.py` measures how much one or two labeled epochs from a held-out session reduce error.
- Duration-dependent targets such as `firing_rate_hz` and `log_firing_rate_hz` may be missing for some exported rows; the trainer filters those rows before fitting and reports the kept counts in its metrics output.
- Current project benchmark candidate: nonlinear LOSO with `movement_summaries`, `population_context`, and `cell_metadata`, mean MAE `0.4265`, mean RMSE `0.5764`.
- Reading `.mat` files requires `scipy`, which is not vendored into this repo.
