import numpy as np
import pytest

from universeframe.fitness import draco_math as dm


def test_fit_log_linear_regression_recovers_uptrend():
    prices = 100.0 * np.exp(np.linspace(0, 0.2, 60))
    fit = dm.fit_log_linear_regression(prices)
    assert fit is not None
    slope, intercept, index, log_prices = fit
    assert slope > 0
    assert len(index) == len(log_prices) == 60


def test_fit_log_linear_regression_rejects_short_or_invalid_series():
    assert dm.fit_log_linear_regression([100.0, 101.0]) is None
    assert dm.fit_log_linear_regression([100.0, -1.0, 102.0]) is None


def test_compute_timeframe_metrics_price_below_line_has_negative_z_score():
    n = 100
    log_price = np.log(100.0) + 0.001 * np.arange(n)
    log_price[-1] -= 0.1  # last price dips well below the fitted line
    metrics = dm.compute_timeframe_metrics("TEST", n, list(np.exp(log_price)))
    assert metrics is not None
    assert metrics.z_score < 0
    assert metrics.current_price < metrics.fitted_price


def test_deviation_points_monotonic_in_negative_z():
    assert dm.deviation_points(-3.0) == 6
    assert dm.deviation_points(-2.25) == 5
    assert dm.deviation_points(0.0) == 1
    assert dm.deviation_points(1.0) == 0


def test_annualized_slope_points_monotonic():
    assert dm.annualized_slope_points(0.5) == 6
    assert dm.annualized_slope_points(0.0) == 1
    assert dm.annualized_slope_points(-0.1) == 0


def test_r_squared_points_monotonic():
    assert dm.r_squared_points(0.9) == 3
    assert dm.r_squared_points(0.65) == 2
    assert dm.r_squared_points(0.4) == 1
    assert dm.r_squared_points(0.1) == 0


def test_evaluate_entry_requirements_rejects_insufficient_timeframes():
    assert dm.evaluate_entry_requirements({}, total_score=100.0, current_price=100.0) is False


def test_select_locked_target_none_when_price_above_every_candidate_line():
    n = 300
    log_price = np.log(100.0) + 0.002 * np.arange(n)  # strictly rising, never dips
    prices = list(np.exp(log_price))
    metrics = dm.compute_all_timeframe_metrics(prices)
    current_price = prices[-1]
    assert dm.select_locked_target(metrics, current_price) is None


def test_project_locked_target_compounds_forward():
    target = dm.LockedTarget(
        label="1Y", lookback=252, slope=0.001, intercept=np.log(100.0), endpoint_index=251,
        r_squared=0.8, entry_price=100.0, entry_target_price=100.0, entry_upside=0.0,
    )
    near = dm.project_locked_target(target, trading_days_held=0)
    far = dm.project_locked_target(target, trading_days_held=100)
    assert far > near > 0
