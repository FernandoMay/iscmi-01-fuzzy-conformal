from src.experiment import expected_calibration_error, run


def test_ece_is_bounded():
    assert 0.0 <= expected_calibration_error([0, 1], [0.1, 0.9]) <= 1.0


def test_reproducibility():
    assert run(seed=7, ensemble_size=3) == run(seed=7, ensemble_size=3)


def test_metrics_schema():
    result = run(seed=7, ensemble_size=3)
    assert result["n_test"] > 0
    assert 0.0 <= result["fuzzy_coverage"] <= 1.0
