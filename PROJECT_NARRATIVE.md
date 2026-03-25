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
- targets: `num_spikes`, `log_num_spikes`

## Baseline Strategy

The first baseline is a simple linear regression with:

- held-out-session evaluation
- target defaulting to `log_num_spikes`
- features from task context, position counts, tetrode/cell context, depth, and spikewidth

The trainer is intended as a smoke-tested baseline, not a final modeling approach.

## Current Baseline Result

Latest Bon run-cell baseline:

- training rows: `1092`
- test rows: `177`
- held-out session: `10`
- target: `log_num_spikes`
- MAE: `1.6936`
- RMSE: `2.1107`

This is the current checkpoint metric to beat.

## Likely Next Steps

- Add richer position-derived features from `pos.data`
- Add per-cell normalized firing targets
- Add track-specific analyses for `TrackA` vs `TrackB`
- Consider models that operate on actual spike-event rows rather than cell aggregates
- Expand from `Bon` to additional animals once the single-animal workflow is stable

## Phase Boundary

This is a reasonable stopping point for the initial baseline phase:

- pipeline is end-to-end
- exports are stable and fast enough to use on Eureka
- the repo now has a reproducible benchmark

The next phase should focus on feature engineering and better evaluation, not more project scaffolding.

## Update Rule

Keep this file current whenever the project meaningfully changes:

- new data understanding
- new tables
- new model choices
- new evaluation results
