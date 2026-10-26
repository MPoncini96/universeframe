from __future__ import annotations

import numpy as np
import pandas as pd

import universeframe.fitness.draco as draco_module
import universeframe.fitness.draco_math as dm
from universeframe.fitness.draco import (
    ROLLING_SCORE_WINDOW_YEARS,
    DracoFitnessRequest,
    _resolve_start_date,
    analyze_draco_fitness,
)

LONGEST_LOOKBACK = max(dm.LOOKBACKS.values())


def _mean_reverting_series(n: int, seed: int) -> pd.Series:
    """Uptrending-but-cyclical log price path so Draco's regression framework
    reliably observes periodic dips below the trend line that go on to
    close, without depending on random luck for every run."""
    index = pd.bdate_range(end="2024-12-01", periods=n)
    t = np.arange(n, dtype=float)
    trend = 0.00035 * t
    cycle = 0.07 * np.sin(2.0 * np.pi * t / 55.0)
    noise = np.random.default_rng(seed).normal(0, 0.003, size=n)
    log_price = np.log(100.0) + trend + cycle + noise
    return pd.Series(np.exp(log_price), index=index, dtype="float64")


def _request(ticker: str = "AAA", **overrides) -> DracoFitnessRequest:
    base = dict(ticker=ticker, start_date="2024-06-01", end_date="2024-12-01")
    base.update(overrides)
    return DracoFitnessRequest(**base)


def _with_mocked_download(prices: pd.Series, fn, *args, **kwargs):
    original = draco_module._download_draco_fitness_prices
    try:
        draco_module._download_draco_fitness_prices = lambda *_args, **_kwargs: prices
        return fn(*args, **kwargs)
    finally:
        draco_module._download_draco_fitness_prices = original


def test_analyze_draco_fitness_reports_ticker_only_scope():
    n = LONGEST_LOOKBACK + 400
    prices = _mean_reverting_series(n, seed=11)
    request = _request()

    result = _with_mocked_download(prices, analyze_draco_fitness, request)

    assert result["algorithm"] == "draco"
    assert result["fitScope"] == "ticker_only"
    assert "universe" not in result["config"]
    assert 0.0 <= result["algorithmFit"]["score"] <= 1.0
    assert 0.0 <= result["algorithmFit"]["confidence"] <= 1.0
    assert result["algorithmFit"]["label"] in {"strong", "good", "moderate", "weak", "poor"}
    assert len(result["timeframeBreakdown"]) > 0


def test_analyze_draco_fitness_walk_forward_date_ranges_do_not_overlap():
    n = LONGEST_LOOKBACK + 400
    prices = _mean_reverting_series(n, seed=13)
    request = _request()

    result = _with_mocked_download(prices, analyze_draco_fitness, request)

    train_range = result["walkForward"]["trainDateRange"]
    test_range = result["walkForward"]["testDateRange"]
    assert train_range is not None and test_range is not None
    assert train_range["end"] < test_range["start"]


def test_analyze_draco_fitness_raises_on_empty_history():
    request = _request()
    empty = pd.Series(dtype="float64")
    try:
        _with_mocked_download(empty, analyze_draco_fitness, request)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_resolve_start_date_honors_an_explicit_override():
    assert _resolve_start_date("2020-01-01", "2026-01-01") == "2020-01-01"


def test_resolve_start_date_rolls_a_fixed_window_before_end_date():
    resolved = _resolve_start_date(None, "2026-10-26")
    expected = (pd.Timestamp("2026-10-26") - pd.Timedelta(days=round(ROLLING_SCORE_WINDOW_YEARS * 365.25))).date().isoformat()
    assert resolved == expected
    resolved_later = _resolve_start_date(None, "2027-10-26")
    span_now = (pd.Timestamp("2026-10-26") - pd.Timestamp(resolved)).days
    span_later = (pd.Timestamp("2027-10-26") - pd.Timestamp(resolved_later)).days
    assert span_now == span_later


def test_analyze_draco_fitness_defaults_to_the_rolling_window_when_dates_are_omitted():
    n = LONGEST_LOOKBACK + 400
    index = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    t = np.arange(n, dtype=float)
    log_price = np.log(100.0) + 0.00035 * t + 0.07 * np.sin(2.0 * np.pi * t / 55.0) + np.random.default_rng(17).normal(0, 0.003, size=n)
    prices = pd.Series(np.exp(log_price), index=index, dtype="float64")
    request = DracoFitnessRequest(ticker="AAA")

    result = _with_mocked_download(prices, analyze_draco_fitness, request)

    start = pd.Timestamp(result["dateRange"]["start"])
    end = pd.Timestamp(result["dateRange"]["end"])
    window_days = (end - start).days
    expected_days = round(ROLLING_SCORE_WINDOW_YEARS * 365.25)
    assert abs(window_days - expected_days) <= 1
