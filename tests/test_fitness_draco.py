from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import universeframe.fitness.draco as draco_module
import universeframe.fitness.draco_math as dm
from universeframe.fitness.common import FIT_SCOPE, build_universe_identity
from universeframe.fitness.draco import (
    DEFAULT_DRACO_UNIVERSE,
    DracoConfig,
    DracoFitnessRequest,
    _analyze_combo,
    _build_request,
    analyze_draco_fitness,
    evaluate_candidate,
    evaluate_position_exit,
    rank_candidates,
)

LONGEST_LOOKBACK = max(dm.LOOKBACKS.values())


def _uptrend_series(n: int, start: float = 100.0, daily_rate: float = 0.001) -> pd.Series:
    index = pd.bdate_range(end="2024-12-01", periods=n)
    return pd.Series(start * np.exp(daily_rate * np.arange(n)), index=index)


def _synthetic_prices(tickers: list[str], n: int, seed: int) -> pd.DataFrame:
    """Uptrending-but-cyclical log price paths so Draco's mean-reversion entry
    conditions (positive long-term slope + periodic dips below the regression
    line) are reliably satisfied without depending on random luck."""
    index = pd.bdate_range(end="2024-12-01", periods=n)
    t = np.arange(n, dtype=float)
    data: dict[str, np.ndarray] = {}
    for i, ticker in enumerate(tickers):
        trend = (0.00035 + 0.00002 * i) * t
        cycle = 0.06 * np.sin(2.0 * np.pi * t / (50.0 + 5.0 * i))
        noise = np.random.default_rng(seed + i).normal(0, 0.003, size=n)
        log_price = np.log(100.0) + trend + cycle + noise
        data[ticker] = np.exp(log_price)
    return pd.DataFrame(data, index=index, dtype=float)


def _request(universe: list[str], **overrides) -> DracoFitnessRequest:
    base = dict(
        ticker="AAA",
        universe=universe,
        market_regime_ticker="SPY",
        fallback_ticker="QQQ",
        min_entry_scores=[10, 15],
        max_positions_options=[2, 3],
        start_date="2024-06-01",
        end_date="2024-12-01",
        forward_days=[1, 5],
        score_mode="blended",
        use_market_regime_filter=False,
        stop_loss_percent=0.10,
        maximum_holding_trading_days=126,
        symbol_reentry_cooldown_days=5,
    )
    base.update(overrides)
    return DracoFitnessRequest(**base)


def test_evaluate_candidate_requires_a_qualifying_trend_and_target():
    cfg = DracoConfig(universe=["AAA"])
    series = _uptrend_series(LONGEST_LOOKBACK + 10, daily_rate=0.0015)
    # A pure uptrend never dips below its own regression line, so it should
    # fail the "price below target line" requirement and return None.
    assert evaluate_candidate("AAA", series, cfg) is None


def test_evaluate_candidate_accepts_a_dip_below_a_strong_uptrend():
    n = LONGEST_LOOKBACK + 10
    t = np.arange(n, dtype=float)
    log_price = np.log(100.0) + 0.0012 * t
    log_price[-5:] -= 0.08  # recent dip below the long-term trend line
    series = pd.Series(np.exp(log_price), index=pd.bdate_range(end="2024-12-01", periods=n))
    cfg = DracoConfig(universe=["AAA"])
    candidate = evaluate_candidate("AAA", series, cfg)
    assert candidate is not None
    assert candidate.target.entry_upside > 0


def test_rank_candidates_orders_by_score_then_r_squared_then_upside():
    def _candidate(ticker, score, r2, upside):
        target = dm.LockedTarget(
            label="1Y", lookback=252, slope=0.001, intercept=4.6, endpoint_index=251,
            r_squared=r2, entry_price=100.0, entry_target_price=100.0 * (1 + upside), entry_upside=upside,
        )
        return draco_module.DracoCandidate(ticker=ticker, total_score=score, metrics={}, target=target)

    candidates = [_candidate("LOW", 20, 0.5, 0.1), _candidate("HIGH", 40, 0.5, 0.1)]
    ranked = rank_candidates(candidates)
    assert [c.ticker for c in ranked] == ["HIGH", "LOW"]


def test_evaluate_position_exit_priority_stop_loss_then_target_then_max_hold():
    # slope=0 keeps the projected target price constant (exp(intercept)==110)
    # regardless of holding_days, isolating each exit branch independently.
    target = dm.LockedTarget(
        label="1Y", lookback=252, slope=0.0, intercept=float(np.log(110.0)), endpoint_index=251,
        r_squared=0.8, entry_price=100.0, entry_target_price=110.0, entry_upside=0.10,
    )
    cfg = DracoConfig(universe=[], stop_loss_percent=0.10, maximum_holding_trading_days=50)

    reason, ret, _proj = evaluate_position_exit(current_price=89.0, entry_price=100.0, holding_days=5, target=target, cfg=cfg)
    assert reason == "stop_loss"
    assert ret == pytest.approx(-0.11)

    reason, _ret, _proj = evaluate_position_exit(current_price=112.0, entry_price=100.0, holding_days=5, target=target, cfg=cfg)
    assert reason == "target_reached"

    reason, _ret, _proj = evaluate_position_exit(current_price=101.0, entry_price=100.0, holding_days=51, target=target, cfg=cfg)
    assert reason == "max_hold"

    reason, _ret, _proj = evaluate_position_exit(current_price=101.0, entry_price=100.0, holding_days=5, target=target, cfg=cfg)
    assert reason is None


def test_draco_combo_tracks_activation_metrics():
    universe = ["AAA", "BBB", "CCC", "SPY"]
    n = LONGEST_LOOKBACK + 260
    prices = _synthetic_prices(universe, n, seed=7)
    request = _request(universe)
    result = _analyze_combo(prices, request, LONGEST_LOOKBACK, min_entry_score=10, max_positions=2)
    assert result["maxPositions"] == 2
    assert result["minimumEntryScore"] == 10
    assert result["eligibleDays"] == (n - LONGEST_LOOKBACK)
    assert 0.0 <= result["comboQuality"] <= 1.0


def test_draco_analyze_reports_universe_conditioned_scope():
    universe = ["AAA", "BBB", "CCC"]
    n = LONGEST_LOOKBACK + 260
    prices = _synthetic_prices([*universe, "SPY"], n, seed=11)
    request = _request(universe)

    original = draco_module._download_draco_fitness_prices
    try:
        draco_module._download_draco_fitness_prices = lambda _req: prices
        result = analyze_draco_fitness(request)
    finally:
        draco_module._download_draco_fitness_prices = original

    assert result["algorithm"] == "draco"
    assert result["fitScope"] == FIT_SCOPE
    assert result["universeIdentity"] == build_universe_identity(universe)
    assert "minimumEntryScore" in result["algorithmFit"]["bestProfile"]
    assert "maxPositionsOptions" in result["algorithmFit"]["recommendedRegion"]
    assert 0.0 <= result["algorithmFit"]["score"] <= 1.0


def test_draco_default_universe_uses_sector_etfs_and_adds_target():
    args = type(
        "Args",
        (),
        {
            "ticker": "NVDA",
            "universe": None,
            "market_regime_ticker": "SPY",
            "fallback_ticker": "QQQ",
            "min_entry_scores": "20,25,30,35,40",
            "max_positions": "2,3,4,5,8",
            "start_date": "2025-01-01",
            "end_date": "2025-02-01",
            "forward_days": "1,5,10",
            "score_mode": "blended",
            "use_market_regime_filter": True,
            "stop_loss_percent": 0.10,
            "maximum_holding_trading_days": 126,
            "symbol_reentry_cooldown_days": 5,
        },
    )()
    request = _build_request(args)
    assert request.universe[:-1] == DEFAULT_DRACO_UNIVERSE
    assert request.universe[-1] == "NVDA"
