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
- The baseline workflow now includes:
  - feature-group ablations
  - ridge alpha sweeps
  - leave-one-session-out evaluation
  - per-session profile summaries for diagnosing hard sessions

## What We Learned From Bon

- Sessions available: `03` through `10`
- Combined Bon export produced:
  - `58` epoch rows
  - `3257` cell rows
- Run epochs include track labels such as `TrackA` and `TrackB`
- Session-to-session variation in active cells and spike counts is substantial, so cross-session evaluation matters.
- Harder held-out sessions are currently `6`, `7`, and especially `9`.

## Current Modeling Shape

The current modeling table is one row per `(session, epoch, tetrode, cell)` during `run` epochs only.

Columns include:

- identifiers: animal, session, epoch, tetrode, cell
- epoch context: environment, exposure, experiment day, position row counts
- cell metadata: depth, spikewidth
- targets: `num_spikes`, `log_num_spikes`, `firing_rate_hz`, `log_firing_rate_hz`, and the diagnostic target `session_centered_log_firing_rate_hz`

Duration-dependent targets are only populated when epoch duration can be computed from `pos.data`.
Movement summaries now also include acceleration-derived features from `vel` and `time`.

## Baseline Strategy

The first baseline is a simple linear regression with:

- held-out-session evaluation
- target now best interpreted as `log_firing_rate_hz`
- standardized numeric features
- ridge regularization
- default features from movement summaries

The trainer is intended as a smoke-tested baseline, not a final modeling approach.
Rows with missing values for the selected target are filtered at training time rather than dropped during export.

## Current Baseline Result

Latest Bon run-cell baseline comparison:

- training rows: `1092`
- test rows: `177`
- held-out session: `10`
- default baseline (`movement_summaries`, alpha `100`): MAE `0.3731`, RMSE `0.6103`
- best MAE in the current alpha sweep (`movement_summaries`, alpha `10`): MAE `0.3618`, RMSE `0.6130`
- best RMSE in the current alpha sweep (`population_context` only, alpha `100`): MAE `0.4040`, RMSE `0.6071`
- adding acceleration features improved the movement-only baseline enough to become the new default
- adding a small nonlinear movement expansion did not materially improve the benchmark

Current default benchmark to beat:

- target: `log_firing_rate_hz`
- feature groups: `movement_summaries`
- ridge alpha: `100`
- MAE: `0.3731`
- RMSE: `0.6103`

## Cross-Session Evaluation

Leave-one-session-out evaluation for the current default baseline:

- mean MAE: `0.3978`
- mean RMSE: `0.6027`
- easiest sessions: `3`, `4`, `8`
- hardest sessions: `6`, `7`, `9`

Session-centered target diagnostics did not materially improve the cross-session result:

- session-centered LOSO mean MAE: `0.3991`
- session-centered LOSO mean RMSE: `0.6040`

Interpretation from the current phase:

- the remaining error is not mainly a session-level offset problem
- harder sessions have higher spike load, denser active populations, and stronger movement intensity
- the current linear ridge model still struggles with that harder regime even after stronger movement features

## Likely Next Steps

- Compare the current linear ridge baseline against a stronger nonlinear model baseline
- Consider track-specific baselines if staying dependency-light
- Add richer position-derived features only when they are motivated by a specific failure mode
- Add track-specific analyses for `TrackA` vs `TrackB`
- Consider models that operate on actual spike-event rows rather than cell aggregates
- Expand from `Bon` to additional animals once the single-animal workflow is stable

The target-selection question is now mostly settled for the current phase:

- use firing-rate-style targets rather than raw spike counts when epoch durations vary

Feature-engineering lesson from the current phase:

- targeted feature-group ablations were informative and changed the benchmark choice
- richer movement summaries, including acceleration features, are the current best default tradeoff
- explicit ablations and alpha sweeps were useful, but further hand-built linear features showed diminishing returns
- the next phase should focus more on model class and evaluation robustness than on piling on more summary features

## Phase Boundary

This is a reasonable stopping point for the initial baseline phase:

- pipeline is end-to-end
- exports are stable and fast enough to use on Eureka
- the repo now has a reproducible benchmark
- the project now has a better default target (`log_firing_rate_hz`), a stronger default feature set, and a clearer picture of where cross-session failures remain

The next phase should focus on stronger models and more robust cross-session evaluation, not more scaffolding.

## Update Rule

Keep this file current whenever the project meaningfully changes:

- new data understanding
- new tables
- new model choices
- new evaluation results
