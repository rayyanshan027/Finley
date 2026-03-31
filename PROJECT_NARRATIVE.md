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
| Dud | `6` | `0.5689` | `0.7665` |
| Eig | `7` | `0.4068` | `0.5692` |
| Fiv | `9` | `0.2543` | `0.3078` |
| Fra | `11` | `0.4082` | `0.5200` |
| Mil | `5` | `0.3937` | `0.5003` |
| Ten | `7` | `0.5678` | `0.7517` |

## Standard Baseline Comparison

XGBoost under the same feature groups and target:

| Animal | Mean MAE | Mean RMSE | Comparison |
| --- | --- | --- | --- |
| Bon | `0.5960` | `0.7520` | much worse than the custom nonlinear model |
| Con | `0.3049` | `0.4479` | better than the custom nonlinear model |
| Cor | `0.5945` | `0.7296` | much worse than the custom nonlinear model |
| Dud | `0.7723` | `0.9358` | worse than the custom nonlinear model |
| Eig | `0.6470` | `0.7952` | much worse than the custom nonlinear model |
| Fiv | `0.3366` | `0.4145` | worse than the custom nonlinear model |
| Fra | `0.4407` | `0.5461` | slightly worse than the custom nonlinear model |
| Mil | `0.5502` | `0.6740` | worse than the custom nonlinear model |
| Ten | `1.2682` | `1.4335` | much worse than the custom nonlinear model |

Interpretation:

- model ranking is not perfectly stable across animals, but the broader pattern is now clear
- across the 9-animal sweep, the custom nonlinear model is the stronger default and beats XGBoost on `8/9` animals
- `Con` remains an important counterexample, so reporting both models is still more honest than pretending one benchmark dominates without exceptions

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
| Dud | `0.7367` | `0.2020` | `0.2196` |
| Fiv | `0.3217` | `0.2061` | `0.2683` |
| Fra | `0.6165` | `0.3410` | `0.4493` |
| Ten | `0.7516` | `0.3576` | `0.4388` |

This pattern transfers across 7 animals tested in adaptation mode.

## What Did Not Work As Well

- simple linear baselines improved only modestly under strict cross-session evaluation
- sparse one-hot within-session unit identity features were high variance and consistently worse than residual correction in the low-data adaptation setting
- adding more adaptation data was not automatically better than using the most relevant adaptation epoch
- the adaptation result is no longer a narrow anecdote from Bon alone; it now holds across a broader subset of animals with very different baseline difficulty

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
- the 9-animal sweep shows that the custom nonlinear model is a robust default rather than a one-animal anecdote
- the 7-animal adaptation sweep shows that the remaining error is also systematically calibratable, not only dataset-specific noise

That combination of data engineering, evaluation design, baseline comparison, and error-driven iteration is the main contribution of the repo.
