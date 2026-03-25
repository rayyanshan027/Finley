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
  - hard-session residual inspection
  - target-clipping diagnostics for testing tail sensitivity
  - epoch-level session adaptation experiments

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
- broader linear feature sets helped some hard sessions but did not remove the dominant repeated-cell failures
- target clipping at the `0.99` training quantile did not materially change hard-session performance
- the best current candidate is now a pure-Python nonlinear forest-style baseline using `movement_summaries`, `population_context`, and `cell_metadata`

Current default benchmark to beat:

- target: `log_firing_rate_hz`
- feature groups: `movement_summaries`, `population_context`, `cell_metadata`
- model: nonlinear forest-style baseline
- leave-one-session-out mean MAE: `0.4265`
- leave-one-session-out mean RMSE: `0.5764`

Simple linear reference benchmark:

- target: `log_firing_rate_hz`
- feature groups: `movement_summaries`
- model: ridge baseline with alpha `100`
- held-out session `10`: MAE `0.3731`, RMSE `0.6103`

## Cross-Session Evaluation

Leave-one-session-out evaluation for the current default baseline:

- mean MAE: `0.3978`
- mean RMSE: `0.6027`
- easiest sessions: `3`, `4`, `8`
- hardest sessions: `6`, `7`, `9`

Leave-one-session-out evaluation for the original nonlinear baseline on the same target and feature group:

- mean MAE: `0.3945`
- mean RMSE: `0.6016`
- easiest sessions: `3`, `4`, `8`
- hardest sessions: `6`, `7`, `9`
- improvement over ridge is real but small, so model nonlinearity alone with movement-only features does not resolve the remaining gap

Session-centered target diagnostics did not materially improve the cross-session result:

- session-centered LOSO mean MAE: `0.3991`
- session-centered LOSO mean RMSE: `0.6040`

Interpretation from the current phase:

- the remaining error is not mainly a session-level offset problem
- harder sessions have higher spike load, denser active populations, and stronger movement intensity
- the current linear ridge model still struggles with that harder regime even after stronger movement features

Track-specific nonlinear evaluation added a more precise picture:

- `TrackB` is worse overall than `TrackA`, with LOSO mean MAE `0.4111` and mean RMSE `0.6108`
- `TrackA` is better overall, with LOSO mean MAE `0.3646` and mean RMSE `0.5793`
- sessions `6`, `7`, and `9` are still hard on both tracks
- `TrackB` amplifies the hard-session pattern rather than fully explaining it
- `session 9 / TrackB` is still the clearest hard-case example
- `session 4 / TrackA` no longer looks like the main diagnostic priority in the nonlinear track-specific view

Current interpretation:

- the main difficulty is not just mixing `TrackA` and `TrackB` in one baseline
- the broader problem still looks like a session-specific regime shift across sessions `6`, `7`, and `9`
- `TrackB` makes that regime shift worse, but the same hard sessions remain difficult on `TrackA`
- harder sessions have much higher population spike load and stronger movement intensity than the easier sessions
- residual inspection shows that a small set of repeated cells dominate the largest errors within sessions `6`, `7`, and `9`
- those repeated-cell failures are not resolved by simple target clipping or by adding more linear feature groups alone
- nonlinear models with population and cell context help materially on the hard sessions, especially session `9`, which points to nonlinear interaction structure rather than a simple session offset or pure outlier problem

Current best nonlinear comparison:

- feature groups: `movement_summaries`, `population_context`, `cell_metadata`
- LOSO mean MAE: `0.4265`
- LOSO mean RMSE: `0.5764`
- hard-session MAE:
  - session `6`: `0.4407`
  - session `7`: `0.4259`
  - session `9`: `0.3906`

Diagnostic lesson from the current phase:

- the dominant failures are repeated high-rate cells, not only broad per-session drift
- weak cell context such as `depth` and `spikewidth` helps only a little in linear models
- richer linear models improve sessions `6` and `7` but can hurt session `9`
- nonlinear modeling plus population and cell context is the first setting that improves the hard regime without relying on target clipping
- the same repeated cells still dominate the nonlinear residuals, but the misses are smaller than under the linear baselines

Session-adaptive comparison:

- one adapted epoch from the held-out session materially improves the hard cases:
  - session `6`: MAE `0.4407` -> `0.3978`
  - session `7`: MAE `0.4259` -> `0.3628`
  - session `9`: MAE `0.3906` -> `0.3859`
- two adapted epochs help more:
  - session `6`: MAE `0.3566`
  - session `7`: MAE `0.3298`
  - session `9`: MAE `0.3503`
- interpretation: a substantial part of the remaining error is session-specific and becomes learnable with modest within-session calibration data

Session-adaptive identity-feature follow-up:

- adding explicit within-session unit-identity one-hot features did not help
- with `0` adapted epochs, identity features are neutral because no held-out-session units are observed yet
- with `1` adapted epoch, identity features are worse on every hard session:
  - session `6`: MAE `0.3978` -> `0.4075`, RMSE `0.5949` -> `0.6361`
  - session `7`: MAE `0.3628` -> `0.4040`, RMSE `0.5535` -> `0.6128`
  - session `9`: MAE `0.3859` -> `0.4123`, RMSE `0.6737` -> `0.7457`
- with `2` adapted epochs, identity features are still worse:
  - session `6`: MAE `0.3566` -> `0.3821`, RMSE `0.6079` -> `0.6245`
  - session `7`: MAE `0.3298` -> `0.3457`, RMSE `0.5360` -> `0.5763`
  - session `9`: MAE `0.3503` -> `0.3720`, RMSE `0.6587` -> `0.7275`
- interpretation: sparse one-hot unit identity is too high-variance for the amount of within-session supervision available in the `1-2` epoch adaptation setting

## Likely Next Steps

- Treat the nonlinear model with `movement_summaries`, `population_context`, and `cell_metadata` as the current benchmark to beat
- Use two benchmark modes going forward:
  - strict LOSO for unseen-session generalization
  - session-adaptive evaluation for settings where one or two labeled epochs are available
- Keep the current adaptive benchmark as the non-identity nonlinear model; explicit one-hot unit identity should not be the default adaptive path
- Next best step: measure unit overlap between adaptation and evaluation epochs, then test a lower-variance within-session correction such as per-unit residual adjustment or shrinkage-style target encoding instead of sparse one-hot identity
- Consider models that operate on actual spike-event rows or finer temporal bins if per-cell epoch aggregates are collapsing too much structure
- Expand from `Bon` to additional animals once the single-animal workflow is stable

The target-selection question is now mostly settled for the current phase:

- use firing-rate-style targets rather than raw spike counts when epoch durations vary

Feature-engineering lesson from the current phase:

- targeted feature-group ablations were informative and changed the benchmark choice
- richer movement summaries, including acceleration features, are the current best default tradeoff
- explicit ablations and alpha sweeps were useful, but further hand-built linear features showed diminishing returns
- simple target clipping did not address the main failure mode
- moving from ridge to a stronger nonlinear baseline with population and cell context gave the first meaningful hard-session improvement
- naive one-hot unit identity in the adaptive setting increased error, so the next phase should focus on lower-variance within-session corrections and unit-overlap diagnostics rather than more sparse identity features

## Phase Boundary

This is a reasonable stopping point for the initial linear-baseline phase:

- pipeline is end-to-end
- exports are stable and fast enough to use on Eureka
- the repo now has a reproducible benchmark
- the project now has a better default target (`log_firing_rate_hz`), a simple ridge reference, a stronger nonlinear benchmark candidate, and a clear session-adaptive evaluation mode
- the remaining problem is now localized well enough to focus on repeated-cell failures and hard-session modeling rather than more pipeline scaffolding

The next phase should focus on session adaptation, unit-aware residual diagnostics, and failure-mode-specific modeling changes, not more scaffolding.

## Update Rule

Keep this file current whenever the project meaningfully changes:

- new data understanding
- new tables
- new model choices
- new evaluation results
