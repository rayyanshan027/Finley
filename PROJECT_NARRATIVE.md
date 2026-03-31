# Project Narrative

## Goal

Build a practical ML workflow for CRCNS HC-6 data that starts from nested MATLAB session files and ends with cross-session predictive benchmarks on run-epoch firing rates.

## Final Scope

The final repo covers four layers of work:

1. dataset inventory and session parsing
2. flattened exports and run-cell modeling table construction
3. strict leave-one-session-out benchmarking
4. post-hoc diagnostics and session adaptation experiments

The modeling table uses one row per `(session, epoch, tetrode, cell)` during `run` epochs only.

## Data Pipeline Outcome

- HC-6 file families are recognized and inventoried across animals and sessions
- session loaders unwrap the day/epoch/tetrode/cell nesting from the MATLAB files
- flattened epoch and cell tables can be exported for inspection
- a run-only model table is built with:
  - identifiers: animal, session, epoch, tetrode, cell
  - epoch context: environment, exposure, experiment day, position-row counts
  - movement summaries: duration, speed, acceleration, movement fraction, spatial range
  - population context
  - cell metadata such as depth and spike width
  - targets including `firing_rate_hz` and `log_firing_rate_hz`

## Canonical Strict LOSO Benchmark

Recommended benchmark setting:

- model: pure-Python nonlinear forest-style baseline
- target: `log_firing_rate_hz`
- feature groups: `movement_summaries`, `population_context`, `cell_metadata`

Strict leave-one-session-out results:

| Animal | Sessions | Mean MAE | Mean RMSE |
| --- | --- | --- | --- |
| Bon | `8` | `0.4265` | `0.5764` |
| Con | `6` | `0.3370` | `0.4848` |
| Cor | `9` | `0.3474` | `0.4500` |

## Standard Baseline Comparison

XGBoost under the same feature groups and target:

| Animal | Mean MAE | Mean RMSE | Comparison |
| --- | --- | --- | --- |
| Bon | `0.5960` | `0.7520` | much worse than the custom nonlinear model |
| Con | `0.3049` | `0.4479` | better than the custom nonlinear model |
| Cor | `0.5945` | `0.7296` | much worse than the custom nonlinear model |

Interpretation:

- model ranking is not stable across animals
- reporting both the custom nonlinear model and a strong library baseline is more honest than presenting one as a universal winner

## Hard Sessions And Residual Structure

The hardest held-out sessions examined in detail were:

- Bon: `6`, `7`, `9`
- Con: `1`, `3`, `4`
- Cor: `1`, `6`, `9`

Residual inspection on Bon showed that the worst misses were not random. The same units repeatedly dominated the largest residuals within hard sessions, which suggested that the remaining generalization gap was partly a unit-level calibration problem rather than only a broad session-level offset.

## Session Adaptation Result

The strongest result in the repo is the adaptation experiment:

- train the shared nonlinear model on all non-held-out sessions
- add one labeled epoch from the held-out session
- fit a per-unit residual correction on that adapted epoch
- apply the learned offsets to later epochs in the same session

Mean hard-session results:

| Animal | Baseline MAE | 1 adapted epoch + residual correction MAE | Mean RMSE after adaptation |
| --- | --- | --- | --- |
| Bon | `0.4191` | `0.1897` | `0.3031` |
| Con | `0.3831` | `0.2470` | `0.3319` |
| Cor | `0.4341` | `0.3265` | `0.4196` |

This pattern transfers across all three animals tested.

## What Did Not Work As Well

- simple linear baselines improved only modestly under strict cross-session evaluation
- sparse one-hot within-session unit identity features were high variance and consistently worse than residual correction in the low-data adaptation setting
- adding more adaptation data was not automatically better than using the most relevant adaptation epoch

## Evidence About Adaptation Source

For Bon, the epoch-level diagnostics show:

- session `6`: pooled first and second adaptation epochs gave the best MAE on the final epoch
- sessions `7` and `9`: the second adaptation epoch alone gave the best MAE on the final epoch

So the defensible conclusion is:

- the choice of adaptation epoch matters
- recency can help
- pooling multiple adaptation epochs is not always optimal

That is stronger and more accurate than claiming one fixed adaptation rule wins everywhere.

## Final Takeaway

The project started as a data-wrangling and baseline-modeling exercise, but the final result is a more useful ML story:

- cross-session prediction on HC-6 is feasible but nontrivial
- strict evaluation reveals substantial session shift
- the remaining error is concentrated in repeated units on hard sessions
- a small amount of within-session supervision plus per-unit residual calibration reduces that error sharply

That combination of data engineering, evaluation design, baseline comparison, and error-driven iteration is the main contribution of the repo.
