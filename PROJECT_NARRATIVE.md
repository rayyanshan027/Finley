# Project Narrative

## Goal

Build a practical ML workflow for CRCNS HC-6 data on Eureka HPC, starting from one animal (`Bon`) and moving from file inventory to modeling-ready tables.

## Current Status

- The repo can inventory HC-6 files and classify modalities including `spikes`, `pos`, `rawpos`, `task`, `cellinfo`, `tetinfo`, and `eeg`.
- The repo can inspect `.mat` contents and unwrap the documented day/epoch/tetrode/cell nesting.
- Flattened exports now exist for:
  - epoch-level tables
  - cell-level spike tables
  - multi-session combined tables for one animal
  - a run-only cell-level modeling table
- A first baseline trainer exists for the run-only model table.

## What We Learned From Bon

- Sessions available: `03` through `10`
- Combined Bon export produced:
  - `58` epoch rows
  - `3257` cell rows
- Run epochs include track labels such as `TrackA` and `TrackB`
- Session-to-session variation in active cells and spike counts is substantial, so cross-session evaluation matters.

## Current Modeling Shape

The current modeling table is one row per `(session, epoch, tetrode, cell)` during `run` epochs only.

Columns include:

- identifiers: animal, session, epoch, tetrode, cell
- epoch context: environment, exposure, experiment day, position row counts
- cell metadata: depth, spikewidth
- targets: `num_spikes`, `log_num_spikes`, `firing_rate_hz`, `log_firing_rate_hz`

Duration-dependent targets are only populated when epoch duration can be computed from `pos.data`.

## Baseline Strategy

The first baseline is a simple linear regression with:

- held-out-session evaluation
- target now best interpreted as `log_firing_rate_hz`
- standardized numeric features
- ridge regularization
- features from task context, position counts, movement summaries, tetrode/cell context, depth, and spikewidth

The trainer is intended as a smoke-tested baseline, not a final modeling approach.
Rows with missing values for the selected target are filtered at training time rather than dropped during export.

## Current Baseline Result

Latest Bon run-cell baseline comparison:

- training rows: `1092`
- test rows: `177`
- held-out session: `10`
- `log_num_spikes`: MAE `1.7028`, RMSE `2.1721`
- `log_firing_rate_hz`: MAE `0.4830`, RMSE `0.6645`
- `log_firing_rate_hz` with richer epoch-level position summaries and stabilized ridge regression: MAE `0.9204`, RMSE `1.0794`

Current default benchmark to beat:

- target: `log_firing_rate_hz`
- MAE: `0.4830`
- RMSE: `0.6645`

## Likely Next Steps

- Add targeted feature-group ablations before adding more features
- Add richer position-derived features from `pos.data`
- Add per-cell normalized firing targets
- Add track-specific analyses for `TrackA` vs `TrackB`
- Consider models that operate on actual spike-event rows rather than cell aggregates
- Expand from `Bon` to additional animals once the single-animal workflow is stable

The target-selection question is now mostly settled for the current phase:

- use firing-rate-style targets rather than raw spike counts when epoch durations vary

Feature-engineering lesson from the current phase:

- adding broader epoch-level movement summaries did not improve the benchmark
- the simpler firing-rate baseline remains stronger than the richer position-summary version
- the next feature work should be more targeted, and ideally evaluated through feature-group ablations

## Phase Boundary

This is a reasonable stopping point for the initial baseline phase:

- pipeline is end-to-end
- exports are stable and fast enough to use on Eureka
- the repo now has a reproducible benchmark
- the project now has a better default target (`log_firing_rate_hz`) for subsequent feature work

The next phase should focus on feature engineering and better evaluation, not more project scaffolding.

## Update Rule

Keep this file current whenever the project meaningfully changes:

- new data understanding
- new tables
- new model choices
- new evaluation results
