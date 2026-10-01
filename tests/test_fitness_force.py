from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import universeframe.fitness.force as force_module
from universeframe.fitness.common import FIT_SCOPE, build_universe_identity
from universeframe.fitness.force import (
    DEFAULT_FORCE_UNIVERSE,
    Alpha1Config,
    ForceFitnessRequest,
    _analyze_combo,
    _build_request,
    analyze_force_fitness,
    choose_holdings_for_day,
    clamp_top_n,
    compute_cost_drag,
    get_trailing_returns,
    normalize_rank_weights,
    rank_with_stability,
)


def test_normalize_rank_weights_sums_to_one():
    weights = normalize_rank_weights([40, 30, 20, 10])
    assert weights.sum() == pytest.approx(1.0)
    assert list(weights) == pytest.approx([0.4, 0.3, 0.2, 0.1])


def test_normalize_rank_weights_falls_back_on_garbage():
    weights = normalize_rank_weights(None)
    assert weights.sum() == pytest.approx(1.0)
    weights = normalize_rank_weights("not json")
    assert weights.sum() == pytest.approx(1.0)


def test_clamp_top_n_respects_weight_count():
    weights = normalize_rank_weights([0.4, 0.3, 0.2, 0.1])
    assert clamp_top_n(10, weights) == 4
    assert clamp_top_n(None, weights) == 4
    assert clamp_top_n(0, weights) == 1


def test_get_trailing_returns_requires_full_window_by_default():
    prices = pd.DataFrame(
        {"AAA": [100.0, 110.0, np.nan, 130.0], "BBB": [100.0, 101.0, 102.0, 103.0]},
    )
    trailing = get_trailing_returns(prices, end_idx_exclusive=4, lookback_days=4)
    # AAA has a gap inside the window and is dropped; BBB has a full window.
    assert "AAA" not in trailing.index
    assert trailing["BBB"] == pytest.approx(0.03)


def test_rank_with_stability_defends_incumbent_within_margin():
    # Incumbent's cushion is 0.05 + 0.15*0.05 = 0.0575 - challenger's 0.054
    # beats the incumbent's raw return but doesn't clear that cushion.
    returns = pd.Series({"INCUMBENT": 0.05, "CHALLENGER": 0.054})
    ranked = rank_with_stability(returns, top_n=1, current_holdings=["INCUMBENT"], margin=0.15)
    assert ranked == ["INCUMBENT"]


def test_rank_with_stability_displaces_incumbent_once_margin_cleared():
    returns = pd.Series({"INCUMBENT": 0.05, "CHALLENGER": 0.20})
    ranked = rank_with_stability(returns, top_n=1, current_holdings=["INCUMBENT"], margin=0.15)
    assert ranked == ["CHALLENGER"]


def test_choose_holdings_falls_back_to_cash_on_kill_switch():
    config = Alpha1Config(universe=["AAA", "BBB"], cash_equivalent="VOO", top_n=1, lookback_days=5)
    all_negative = pd.Series({"AAA": -0.05, "BBB": -0.02})
    selected, weights, risk_off, reason = choose_holdings_for_day(config, all_negative)
    assert risk_off is True
    assert selected == ["VOO"]
    assert reason.startswith("kill_switch:")


def test_choose_holdings_selects_top_n_risk_on():
    config = Alpha1Config(universe=["AAA", "BBB", "CCC"], cash_equivalent="VOO", top_n=2, lookback_days=5)
    returns = pd.Series({"AAA": 0.05, "BBB": 0.03, "CCC": 0.01})
    selected, weights, risk_off, _reason = choose_holdings_for_day(config, returns)
    assert risk_off is False
    assert selected == ["AAA", "BBB"]
    assert weights.sum() == pytest.approx(1.0)


def test_compute_cost_drag_matches_bps_formula():
    assert compute_cost_drag(1.0, transaction_cost_bps=5.0, slippage_bps=5.0) == pytest.approx(0.001)


def _request(**overrides) -> ForceFitnessRequest:
    base = dict(
        ticker="AAA",
        universe=["AAA", "BBB", "CCC"],
        safety_net_equity="VOO",
        lookbacks=[2, 3],
        portfolio_sizes=[1, 2],
        weights=[0.6, 0.4],
        start_date="2025-01-01",
        end_date="2025-02-01",
        forward_days=[1, 3],
        score_mode="blended",
    )
    base.update(overrides)
    return ForceFitnessRequest(**base)


def test_force_combo_tracks_activation_metrics():
    prices = pd.DataFrame(
        {
            "AAA": [100.0, 102.0, 104.0, 106.0, 108.0, 110.0],
            "BBB": [100.0, 101.0, 100.0, 99.0, 98.0, 97.0],
            "CCC": [100.0, 99.0, 98.0, 97.0, 96.0, 95.0],
            "VOO": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        },
        index=pd.date_range("2025-01-01", periods=6, freq="D"),
        dtype=float,
    )
    result = _analyze_combo(prices, _request(), lookback=2, portfolio_size=1)
    assert result["portfolioSize"] == 1
    assert result["activationCount"] > 0
    assert 0.0 <= result["comboQuality"] <= 1.0
    # AAA leads every trailing window in this monotonic uptrend, so it's a
    # single continuous activation run.
    assert result["activationEpisodeCount"] == 1
    assert result["activationCount"] > result["activationEpisodeCount"]


def test_force_analyze_reports_universe_conditioned_scope():
    request = _request()
    original = force_module._build_price_frame
    try:
        force_module._build_price_frame = lambda *_args, **_kwargs: pd.DataFrame(
            {
                "AAA": [100.0, 102.0, 104.0, 106.0, 108.0],
                "BBB": [100.0, 99.0, 98.0, 97.0, 96.0],
                "CCC": [100.0, 101.0, 102.0, 101.0, 100.0],
                "VOO": [100.0, 100.0, 100.0, 100.0, 100.0],
            },
            index=pd.date_range("2025-01-01", periods=5, freq="D"),
            dtype=float,
        )
        result = analyze_force_fitness(request)
    finally:
        force_module._build_price_frame = original

    assert result["algorithm"] == "force"
    assert result["fitScope"] == FIT_SCOPE
    assert result["universeIdentity"] == build_universe_identity(["AAA", "BBB", "CCC"])
    assert "portfolioSize" in result["algorithmFit"]["bestProfile"]
    assert "portfolioSizes" in result["algorithmFit"]["recommendedRegion"]
    assert 0.0 <= result["algorithmFit"]["score"] <= 1.0


def test_force_default_universe_uses_sector_etfs_and_adds_target():
    args = type(
        "Args",
        (),
        {
            "ticker": "NVDA",
            "universe": None,
            "safety_net_equity": "VOO",
            "lookbacks": "14,21,30,45,63",
            "portfolio_sizes": "2,3,4,5",
            "start_date": "2025-01-01",
            "end_date": "2025-02-01",
            "forward_days": "1,3",
            "score_mode": "blended",
        },
    )()
    request = _build_request(args)
    assert request.universe[:-1] == DEFAULT_FORCE_UNIVERSE
    assert request.universe[-1] == "NVDA"


def test_analyze_force_populates_lookback_quality_by_label():
    """When the tested grid includes a LOOKBACK_LABELS lookback (10 here),
    lookbackQualityByLabel must carry a real, out-of-sample value for it -
    the field the force_10d/force_15d/force_45d/force_3m adaptive-universe
    variants read."""
    n = 60
    prices = pd.DataFrame(
        {
            "AAA": [100.0 + i for i in range(n)],
            "BBB": [100.0 - 0.2 * i for i in range(n)],
            "CCC": [100.0] * n,
            "VOO": [100.0] * n,
        },
        index=pd.date_range("2025-01-01", periods=n, freq="D"),
        dtype=float,
    )
    request = _request(lookbacks=[10, 21], forward_days=[1])

    original = force_module._build_price_frame
    try:
        force_module._build_price_frame = lambda *_args, **_kwargs: prices
        result = analyze_force_fitness(request)
    finally:
        force_module._build_price_frame = original

    assert "10d" in result["lookbackQualityByLabel"]
    assert 0.0 <= result["lookbackQualityByLabel"]["10d"] <= 1.0
    assert set(result["lookbackQualityByLabel"].keys()) == {"10d"}
