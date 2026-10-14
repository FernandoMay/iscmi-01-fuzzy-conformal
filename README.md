# SD-01 — Fuzzy-Calibrated Conformal Prediction for Adaptive Uncertainty-Aware AI

ISCMI 2026 candidate package.

The method combines a bootstrap ensemble, an uncertainty score, and a fuzzy-scaled conformal interval. The current implementation is a transparent methodological baseline. It does not claim clinical deployment or diagnostic validity.

## Contents

| Path | Purpose |
| --- | --- |
| `src/experiment.py` | Base binary baseline: fuzzy-scaled conformal intervals and calibration metrics. |
| `src/agent_selection.py` | A-09 extension: fuzzy-calibrated conformal prediction for **safe action selection** with abstention. |
| `data/metrics.json` | Recorded metrics for the base baseline. |
| `data/agent_selection_metrics.json` | Recorded metrics for the action-selection extension. |
| `RESULTS.md` | Recorded numbers, interpretation, and honest limitations. |
| `tests/` | Regression tests for both modules. |

## SD-01 / A-09 — Safe action selection (multi-class extension)

`src/agent_selection.py` generalises the fuzzy-scaled conformal construction from
binary coverage to a finite **action set**: an autonomous agent executes one
action when the conformal prediction set is a singleton and **abstains** otherwise
(the set is empty or ambiguous). It reports marginal coverage, mean set size,
abstention rate, selective risk, and the risk-coverage curve (AURC) on public
scikit-learn proxy datasets (`digits`, `wine`, `iris`). See `RESULTS.md` for the
recorded numbers and their limitations. These datasets are methodological
proxies; no deployment or safety claim is made.
