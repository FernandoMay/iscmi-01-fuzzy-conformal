# Results — SD-01 / A-09

Recorded output of `python src/agent_selection.py` (seed `20260915`,
ensemble size `7`, `alpha = 0.10`, `beta = 0.50`, scikit-learn `1.6.1`).

These are the numbers written to `data/agent_selection_metrics.json`. They are
reproducible for the fixed seed and hyperparameters in `REPRODUCE.md`.

## Protocol

- **Task**: an autonomous agent selects one action from a finite action set, or
  abstains. Public scikit-learn datasets are used as proxies for that action set
  (`digits` = 10 actions, `wine` = 3 actions, `iris` = 3 actions).
- **Method**: split conformal prediction over a bootstrap classifier ensemble.
  The prediction set for a test instance is `{k : 1 - p_k <= tau(x)}`, where
  `p_k` is the ensemble-mean action probability and `tau` is the conformal
  quantile `q`.
  - *baseline conformal*: `tau(x) = q`
  - *fuzzy conformal*: `tau(x) = q * (1 + beta * fuzzy(x))`, with `fuzzy(x)` in
    `[0, 1]` growing with ensemble disagreement.
- **Policy**: execute the action iff the prediction set is a singleton; otherwise
  abstain.
- **Metrics**: marginal coverage, mean set size, abstention rate, selective risk
  (error among executed actions), and the risk-coverage curve / AURC under
  top-action confidence ranking.

## Numbers

| dataset | K | n_test | q | base cov | base \|S\| | base abst | base selRisk | fuzzy cov | fuzzy \|S\| | fuzzy abst | fuzzy selRisk | always-act risk | AURC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| digits | 10 | 360 | 0.1747 | 0.867 | 0.872 | 0.128 | 0.006 | 0.897 | 0.914 | 0.086 | 0.018 | 0.042 | 0.003 |
| wine | 3 | 36 | 0.3308 | 0.944 | 0.944 | 0.056 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| iris | 3 | 30 | 0.4340 | 0.867 | 0.933 | 0.067 | 0.071 | 0.967 | 1.133 | 0.133 | 0.038 | 0.100 | 0.014 |

## Observations

- **Coverage tracks the nominal level.** Baseline conformal coverage is close to
  the target `1 - alpha = 0.90` on all three proxies (0.867, 0.944, 0.867).
- **Fuzzy scaling is conservative.** It raises coverage (0.897, 1.000, 0.967) and
  enlarges prediction sets (`digits` 0.872 -> 0.914, `iris` 0.933 -> 1.133), i.e.
  it trades set size for robustness. On `digits` it also changed the abstention
  rate (0.128 -> 0.086): the baseline produced several *empty* sets (which also
  count as abstention), and fuzzy scaling filled some of them back to a singleton.
  Mean set size rising while abstention falls is therefore consistent, not a bug.
- **Selective risk is low** when the agent acts (0.006 / 0.000 / 0.071 baseline;
  0.018 / 0.000 / 0.038 fuzzy), so abstention concentrates on genuinely ambiguous
  instances.
- **`wine` is not informative** here: it is near-separable (always-act risk 0.000),
  so all policies look perfect. Reported for completeness, not as evidence.
- **AURC is near zero** on all proxies; these tasks are easy, so confidence
  ranking is close to ideal.

## Limitations (read before citing)

- Public *classification* datasets are **proxies**, not autonomous-agent
  deployments and not safety validation. No action has real consequences here.
- `digits`, `iris`, `wine` are small and largely separable; they under-represent
  the ambiguity a real agent faces.
- A single seed is reported. Multi-seed confidence intervals are not yet computed.
- Baselines are limited to (i) non-conformal confidence ranking (AURC) and
  (ii) the non-fuzzy conformal variant. No comparison against other selective
  methods (e.g. softmax-response, MC-dropout, temperature scaling) yet.
- No time-series or distribution-shift evaluation yet; conformal coverage assumes
  exchangeability.

These gaps are tracked in the backlog record for A-09 and must be closed before
any submission-grade claim.
