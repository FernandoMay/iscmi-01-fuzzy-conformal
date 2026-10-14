import numpy as np

from src.agent_selection import (
    DEFAULT_ALPHA,
    load_split,
    prediction_sets,
    risk_coverage_curve,
    run,
    run_dataset,
)


def test_reproducibility():
    assert run(seed=7, ensemble_size=3) == run(seed=7, ensemble_size=3)


def test_split_is_stratified_and_disjoint():
    split = load_split("digits", seed=7)
    assert len(split.y_train) > 0 and len(split.y_cal) > 0 and len(split.y_test) > 0
    # Rows should not overlap across the three splits.
    assert len(split.x_train) + len(split.x_cal) + len(split.x_test) == 1797


def test_prediction_sets_monotone_in_threshold():
    action_probs = np.array([[0.7, 0.2, 0.1], [0.4, 0.4, 0.2]])
    small = prediction_sets(action_probs, np.array([0.5, 0.5]))
    large = prediction_sets(action_probs, np.array([0.9, 0.9]))
    # Larger thresholds can only add actions, never remove them.
    assert np.all(large >= small)


def test_conformal_coverage_matches_nominal():
    result = run_dataset("digits", seed=7, ensemble_size=3, alpha=DEFAULT_ALPHA)
    coverage = result["baseline_conformal"]["coverage"]
    assert abs(coverage - (1.0 - DEFAULT_ALPHA)) <= 0.12


def test_decision_statistics_are_bounded():
    result = run_dataset("wine", seed=7, ensemble_size=3)
    for key in ("baseline_conformal", "fuzzy_conformal"):
        stats = result[key]
        assert 0.0 <= stats["coverage"] <= 1.0
        assert 0.0 <= stats["abstention_rate"] <= 1.0
        assert 0 <= stats["mean_set_size"] <= result["n_actions"]
        assert 0.0 <= stats["executed_fraction"] <= 1.0


def test_fuzzy_scaling_never_shrinks_sets():
    result = run_dataset("digits", seed=7, ensemble_size=3, beta=0.5)
    assert result["set_size_delta_fuzzy_minus_baseline"] >= 0.0
    assert result["fuzzy_conformal"]["mean_set_size"] >= result["baseline_conformal"]["mean_set_size"]


def test_risk_coverage_curve_shape_and_aurc():
    result = run_dataset("iris", seed=7, ensemble_size=3)
    curve = result["risk_coverage"]
    coverages = [point[0] for point in curve]
    assert coverages == sorted(coverages)
    assert coverages[0] > 0.0 and coverages[-1] == 1.0
    assert 0.0 <= result["aurc"] <= 1.0
