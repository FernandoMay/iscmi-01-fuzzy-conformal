"""A-09 / SD-01 extension: fuzzy-calibrated conformal prediction for safe
action selection in autonomous agents.

The agent must select one action from a finite action set, or ABSTAIN when the
conformal prediction set is not a singleton. We reuse the fuzzy-scaled conformal
construction from ``src/experiment.py`` and generalise it from binary coverage to
multi-class action selection.

Method (split conformal, generalised to K actions):
  * Ensemble of classifiers -> per-action probability distribution.
  * Nonconformity score for the true action s_i = 1 - p_hat(y_i) on a calibration
    split; the conformal quantile q gives finite-sample marginal coverage 1-alpha.
  * Prediction set for a test instance: {k : 1 - p_k <= tau(x)}.
      - baseline conformal: tau(x) = q
      - fuzzy conformal:    tau(x) = q * (1 + beta * fuzzy(x)), where fuzzy(x)
        in [0, 1] grows with ensemble disagreement. Higher uncertainty enlarges
        the set (more conservative), trading set size for robustness.
  * Policy: execute the action iff the set is a singleton; otherwise abstain.

Datasets are public sklearn datasets used only as action-selection proxies.
No deployment or safety claim is made; this is a transparent methodological
baseline.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import sklearn
from sklearn.datasets import load_digits, load_iris, load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_SEED = 20260915
DEFAULT_ENSEMBLE_SIZE = 7
DEFAULT_ALPHA = 0.1
DEFAULT_BETA = 0.5
EPSILON = 1e-9

# Public, license-clean datasets that ship with scikit-learn. Each is used only
# as a proxy for a finite autonomous-agent action set.
DATASETS = {
    "digits": load_digits,  # 10 actions
    "wine": load_wine,  # 3 actions
    "iris": load_iris,  # 3 actions
}


@dataclass
class Split:
    x_train: np.ndarray
    x_cal: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_cal: np.ndarray
    y_test: np.ndarray
    actions: np.ndarray  # canonical action (class) order


def load_split(name: str, seed: int = DEFAULT_SEED) -> Split:
    """Stratified train / calibration / test split for one proxy dataset."""
    loader = DATASETS[name]
    dataset = loader()
    x, y = dataset.data, dataset.target
    actions = np.unique(y)
    x_train, x_other, y_train, y_other = train_test_split(
        x, y, test_size=0.4, stratify=y, random_state=seed
    )
    x_cal, x_test, y_cal, y_test = train_test_split(
        x_other, y_other, test_size=0.5, stratify=y_other, random_state=seed
    )
    return Split(x_train, x_cal, x_test, y_train, y_cal, y_test, actions)


def _aligned_proba(clf, x: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """predict_proba mapped onto a fixed action order, so members are comparable."""
    raw = clf.predict_proba(x)
    aligned = np.zeros((x.shape[0], len(actions)))
    for column, action in enumerate(clf.classes_):
        aligned[:, np.where(actions == action)[0][0]] = raw[:, column]
    return aligned


def ensemble_probabilities(
    split: Split, ensemble_size: int, seed: int
) -> Dict[str, np.ndarray]:
    """Bootstrap ensemble -> (members, rows, actions) probability tensors."""
    rng = np.random.default_rng(seed)
    cal, test = [], []
    for member in range(ensemble_size):
        indices = rng.integers(0, len(split.x_train), len(split.x_train))
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=seed + member + 1),
        )
        clf.fit(split.x_train[indices], split.y_train[indices])
        cal.append(_aligned_proba(clf, split.x_cal, split.actions))
        test.append(_aligned_proba(clf, split.x_test, split.actions))
    return {"calibration": np.asarray(cal), "test": np.asarray(test)}


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal quantile, clipped to the score range [0, 1]."""
    n = len(scores)
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    quantile = float(np.quantile(scores, min(rank, n) / n))
    return float(np.clip(quantile, 0.0, 1.0))


def fuzzy_disagreement(member_probs: np.ndarray, action_probs: np.ndarray) -> np.ndarray:
    """Per-instance uncertainty in [0, 1] from ensemble disagreement."""
    return np.mean(member_probs.std(axis=0), axis=1)


def prediction_sets(
    action_probs: np.ndarray, threshold: np.ndarray
) -> np.ndarray:
    """Boolean (rows, actions) matrix; action k included iff 1 - p_k <= tau."""
    scores = 1.0 - action_probs
    return scores <= threshold[:, None]


def _decision_stats(
    sets: np.ndarray, y_test: np.ndarray, actions: np.ndarray
) -> Dict[str, float]:
    size = sets.sum(axis=1)
    covered = sets[np.arange(len(y_test)), np.searchsorted(actions, y_test)]
    acted = size == 1
    correct_when_acted = covered & acted
    n = len(y_test)
    n_acted = int(acted.sum())
    return {
        "coverage": float(covered.mean()),
        "mean_set_size": float(size.mean()),
        "abstention_rate": float(1.0 - acted.mean()),
        "selective_accuracy": float(correct_when_acted.sum() / n_acted) if n_acted else float("nan"),
        "selective_risk": float(1.0 - correct_when_acted.sum() / n_acted) if n_acted else float("nan"),
        "executed_fraction": float(acted.mean()),
    }


def risk_coverage_curve(
    action_probs: np.ndarray,
    member_probs: np.ndarray,
    y_test: np.ndarray,
    actions: np.ndarray,
    points: int = 20,
) -> List[List[float]]:
    """Selective-prediction risk-coverage curve using top-action confidence.

    Instances are ranked by confidence (max action probability). At coverage c
    the risk is the error rate among the c most confident instances.
    """
    confidence = action_probs.max(axis=1)
    errors = (action_probs.argmax(axis=1) != np.searchsorted(actions, y_test)).astype(float)
    order = np.argsort(-confidence)
    errors = errors[order]
    n = len(errors)
    curve = []
    cumulative = np.cumsum(errors)
    for i in range(1, points + 1):
        k = max(1, int(round(n * i / points)))
        coverage = k / n
        risk = float(cumulative[k - 1] / k)
        curve.append([float(coverage), risk])
    return curve


def aurc(curve: List[List[float]]) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    if len(curve) < 2:
        return float("nan")
    array = np.asarray(curve)
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return float(integrate(array[:, 1], array[:, 0]))


def run_dataset(
    name: str,
    seed: int = DEFAULT_SEED,
    ensemble_size: int = DEFAULT_ENSEMBLE_SIZE,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
) -> Dict[str, object]:
    split = load_split(name, seed)
    probs = ensemble_probabilities(split, ensemble_size, seed)

    cal_action = probs["calibration"].mean(axis=0)
    test_action = probs["test"].mean(axis=0)
    test_members = probs["test"]

    true_action = np.searchsorted(split.actions, split.y_cal)
    scores = 1.0 - cal_action[np.arange(len(true_action)), true_action]
    q = conformal_quantile(scores, alpha)

    cal_disagreement = fuzzy_disagreement(probs["calibration"], cal_action)
    scale = float(np.quantile(cal_disagreement, 0.75)) + EPSILON
    fuzzy = np.clip(fuzzy_disagreement(test_members, test_action) / scale, 0.0, 1.0)

    base_threshold = np.full(len(fuzzy), q)
    fuzzy_threshold = np.clip(q * (1.0 + beta * fuzzy), 0.0, 1.0)

    base_stats = _decision_stats(prediction_sets(test_action, base_threshold), split.y_test, split.actions)
    fuzzy_stats = _decision_stats(prediction_sets(test_action, fuzzy_threshold), split.y_test, split.actions)

    always_act_accuracy = float(
        accuracy_score(split.y_test, split.actions[test_action.argmax(axis=1)])
    )
    curve = risk_coverage_curve(test_action, test_members, split.y_test, split.actions)

    return {
        "n_train": len(split.y_train),
        "n_calibration": len(split.y_cal),
        "n_test": len(split.y_test),
        "n_actions": int(len(split.actions)),
        "conformal_quantile": q,
        "always_act_accuracy": always_act_accuracy,
        "always_act_risk": float(1.0 - always_act_accuracy),
        "aurc": aurc(curve),
        "risk_coverage": curve,
        "baseline_conformal": base_stats,
        "fuzzy_conformal": fuzzy_stats,
        "set_size_delta_fuzzy_minus_baseline": float(
            fuzzy_stats["mean_set_size"] - base_stats["mean_set_size"]
        ),
    }


def run(
    seed: int = DEFAULT_SEED,
    ensemble_size: int = DEFAULT_ENSEMBLE_SIZE,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
) -> Dict[str, object]:
    return {
        "provenance": {
            "method": "fuzzy-calibrated split conformal prediction for selective action selection",
            "datasets": "sklearn public proxies (digits, wine, iris); not a deployment or safety validation",
            "sklearn_version": sklearn.__version__,
        },
        "seed": seed,
        "ensemble_size": ensemble_size,
        "alpha": alpha,
        "beta": beta,
        "datasets": {
            name: run_dataset(name, seed, ensemble_size, alpha, beta)
            for name in DATASETS
        },
    }


if __name__ == "__main__":
    output = run()
    Path("data").mkdir(exist_ok=True)
    Path("data/agent_selection_metrics.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))
