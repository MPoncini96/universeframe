import math

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


# ---------------------------------------------------------------------------
# Mean-reversion fitness scoring
# ---------------------------------------------------------------------------

def _perfect_exponential_series(n: int, daily_log_return: float, start_price: float = 100.0) -> list[float]:
    return [start_price * math.exp(daily_log_return * i) for i in range(n)]


def test_find_dip_episodes_scores_only_the_first_day_of_each_run():
    z_scores = [0.5, -1.2, -1.5, -1.1, 0.2, -1.3, -1.4]
    fitted = [100.0] * 7
    full_prices = [99.0, 95.0, 94.0, 94.0, 102.0, 90.0, 100.0, 100.0, 80.0, 100.0]

    episodes = dm.find_dip_episodes(z_scores, fitted, full_prices, anchor_offset=0, horizon_days=2, fill_lag_days=1)

    assert [e.anchor_index for e in episodes] == [1, 5]
    assert episodes[0].reverted is True
    assert episodes[1].reverted is False


def test_find_dip_episodes_leaves_outcome_unjudged_without_enough_forward_data():
    episodes = dm.find_dip_episodes([-1.5], [100.0], [90.0], anchor_offset=5, horizon_days=2, fill_lag_days=1)
    assert len(episodes) == 1
    assert episodes[0].reverted is None


def test_summarize_reversion_excludes_unjudged_episodes():
    episodes = [
        dm.DipEpisode(anchor_index=0, z_score=-1.2, fitted_price_at_anchor=100.0, reverted=True),
        dm.DipEpisode(anchor_index=5, z_score=-1.3, fitted_price_at_anchor=100.0, reverted=False),
        dm.DipEpisode(anchor_index=9, z_score=-1.4, fitted_price_at_anchor=100.0, reverted=None),
    ]
    stats = dm.summarize_reversion("1M", episodes)
    assert stats.episode_count == 2
    assert stats.reverted_count == 1
    assert stats.reversion_probability == 0.5


def test_fit_label_thresholds():
    assert dm.fit_label(0.85) == "strong"
    assert dm.fit_label(0.70) == "good"
    assert dm.fit_label(0.55) == "moderate"
    assert dm.fit_label(0.40) == "weak"
    assert dm.fit_label(0.10) == "poor"


def test_normalize_reversion_quality_is_monotonic_and_bounded():
    low = dm.normalize_reversion_quality(0.0)
    mid = dm.normalize_reversion_quality(dm.REVERSION_QUALITY_CENTER)
    high = dm.normalize_reversion_quality(5.0)
    assert 0.0 <= low < mid < high <= 1.0
    assert mid == pytest.approx(0.5, abs=1e-9)


def test_compute_mean_reversion_fitness_none_with_insufficient_history():
    prices = np.array(_perfect_exponential_series(100, daily_log_return=0.001))
    assert dm.compute_mean_reversion_fitness(prices, "AAA") is None


def test_compute_mean_reversion_fitness_zero_for_pure_uptrend_with_no_current_dip():
    n = max(dm.LOOKBACKS.values()) + 300
    prices = np.array(_perfect_exponential_series(n, daily_log_return=0.001))
    result = dm.compute_mean_reversion_fitness(prices, "AAA")
    assert result is not None
    assert result.total_score == 0.0
    assert all(p.expected_value == 0.0 for p in result.profiles)


def test_compute_mean_reversion_fitness_rewards_a_ticker_currently_dipped_with_a_history_of_reverting():
    n = max(dm.LOOKBACKS.values()) + 400
    t = np.arange(n, dtype=float)
    cycle = 0.09 * np.sin(2.0 * np.pi * t / 55.0)
    log_price = math.log(100.0) + 0.00035 * t + cycle
    log_price[-3:] -= 0.09  # force a deep dip on the most recent days
    prices = np.exp(log_price)

    result = dm.compute_mean_reversion_fitness(prices, "AAA")
    assert result is not None
    assert result.total_score > 0.0
    dipped = [p for p in result.profiles if p.z_score_now < 0.0]
    assert dipped
    assert any(p.expected_value > 0.0 for p in dipped)
    assert 0.0 <= result.score <= 1.0
    assert result.label in {"strong", "good", "moderate", "weak", "poor"}
