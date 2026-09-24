"""SD-01: fuzzy-calibrated conformal prediction baseline.

The sklearn breast-cancer dataset is used only as a public binary-classification
proxy. It is not a clinical validation dataset and no diagnostic claim is made.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class Split:
    x_train: np.ndarray
    x_cal: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_cal: np.ndarray
    y_test: np.ndarray


def load_split(seed: int = 20260915) -> Split:
    data = load_breast_cancer()
    x_train, x_other, y_train, y_other = train_test_split(
        data.data, data.target, test_size=0.4, stratify=data.target, random_state=seed
    )
    x_cal, x_test, y_cal, y_test = train_test_split(
        x_other, y_other, test_size=0.5, stratify=y_other, random_state=seed
    )
    return Split(x_train, x_cal, x_test, y_train, y_cal, y_test)


def model(seed: int) -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=seed),
    )


def expected_calibration_error(y_true, probability, bins=10):
    y_true = np.asarray(y_true)
    probability = np.asarray(probability)
    edges = np.linspace(0, 1, bins + 1)
    total = len(y_true)
    error = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (probability >= left) & (probability < right)
        if not np.any(mask):
            continue
        error += np.sum(mask) / total * abs(
            probability[mask].mean() - y_true[mask].mean()
        )
    return float(error)


def run(seed: int = 20260915, ensemble_size: int = 7, alpha: float = 0.1):
    split = load_split(seed)
    rng = np.random.default_rng(seed)

    baseline = model(seed).fit(split.x_train, split.y_train)
    baseline_probability = baseline.predict_proba(split.x_test)[:, 1]

    probabilities = []
    for member in range(ensemble_size):
        indices = rng.integers(0, len(split.x_train), len(split.x_train))
        fitted = model(seed + member + 1).fit(
            split.x_train[indices], split.y_train[indices]
        )
        probabilities.append(fitted.predict_proba(split.x_cal)[:, 1])
    calibration_members = np.asarray(probabilities)

    test_members = []
    for member in range(ensemble_size):
        indices = rng.integers(0, len(split.x_train), len(split.x_train))
        fitted = model(seed + 100 + member).fit(
            split.x_train[indices], split.y_train[indices]
        )
        test_members.append(fitted.predict_proba(split.x_test)[:, 1])
    test_members = np.asarray(test_members)

    calibration_mean = calibration_members.mean(axis=0)
    calibration_uncertainty = calibration_members.std(axis=0)
    scores = np.abs(split.y_cal - calibration_mean)
    conformal_quantile = float(np.quantile(scores, np.ceil((len(scores) + 1) * (1 - alpha)) / len(scores)))

    test_mean = test_members.mean(axis=0)
    test_uncertainty = test_members.std(axis=0)
    uncertainty_scale = np.quantile(calibration_uncertainty, 0.75) + 1e-9
    fuzzy_high = np.clip(test_uncertainty / uncertainty_scale, 0, 1)
    radius = conformal_quantile * (1.0 + 0.5 * fuzzy_high)
    lower = np.clip(test_mean - radius, 0, 1)
    upper = np.clip(test_mean + radius, 0, 1)
    covered = (split.y_test >= lower) & (split.y_test <= upper)

    return {
        "dataset": "sklearn breast_cancer proxy; not clinical validation",
        "seed": seed,
        "ensemble_size": ensemble_size,
        "alpha": alpha,
        "n_train": len(split.y_train),
        "n_calibration": len(split.y_cal),
        "n_test": len(split.y_test),
        "baseline_accuracy": float(accuracy_score(split.y_test, baseline_probability >= 0.5)),
        "baseline_brier": float(brier_score_loss(split.y_test, baseline_probability)),
        "baseline_ece": expected_calibration_error(split.y_test, baseline_probability),
        "conformal_quantile": conformal_quantile,
        "fuzzy_coverage": float(covered.mean()),
        "fuzzy_mean_interval_width": float(np.mean(upper - lower)),
        "fuzzy_ece": expected_calibration_error(split.y_test, test_mean),
    }


if __name__ == "__main__":
    output = run()
    Path("data").mkdir(exist_ok=True)
    Path("data/metrics.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))
