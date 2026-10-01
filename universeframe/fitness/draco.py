"""Draco fitness scoring - a faithful, standalone port of Monstra's
``bots/draco.py`` candidate-evaluation / exit logic plus
``scripts/analyze_draco_stock_fitness.py``'s walk-forward grid search.

Draco is a multi-timeframe log-linear regression mean-reversion strategy
(see ``draco_math.py``): each week it scans a universe for candidates
trading below a strong upward trend line, locks a regression-projected exit
target at entry, and exits on stop loss / target reached / max holding
period / risk-off liquidation. Fitness asks: if a given ticker had been in
Draco's candidate pool this whole time, how good a Draco position would it
actually have been, across a grid of (minimum entry score, max positions)
combos - validated out-of-sample.

What's intentionally left out, relative to the live bot: persisted state
(``trading.draco_state``), the portfolio-level circuit breaker (reads live
account equity), and DB config loading - a standalone fitness run always
starts from a fixed universe and empty state, same as the production
fitness analyzer itself does.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from . import draco_math as dm
from .common import (
    build_universe_identity,
    calculate_algorithm_fit,
    calculate_combo_sample_confidence,
    clean_ticker,
    compute_walk_forward_split_index,
    date_bounds,
    forward_return,
    max_drawdown,
    normalize_combo_quality,
    score_combo,
    FIT_SCOPE,
    WALK_FORWARD_TRAIN_FRACTION,
)
from .prices import download_adjusted_close

DEFAULT_MARKET_REGIME_TICKER = "SPY"
DEFAULT_FALLBACK_TICKER = "QQQ"
DEFAULT_MARKET_SMA_PERIOD = 200
DEFAULT_REGIME_CONFIRMATION_DAYS = 5
DEFAULT_STOP_LOSS_PERCENT = 0.10
DEFAULT_MAXIMUM_HOLDING_TRADING_DAYS = 126
DEFAULT_SYMBOL_REENTRY_COOLDOWN_DAYS = 5

DEFAULT_DRACO_UNIVERSE = [
    "XLK", "XLF", "XLV", "XLY", "XLI", "XLP", "XLE", "XLB", "XLU", "XLRE", "XLC", "SMH",
]
DEFAULT_MIN_ENTRY_SCORES = [20, 25, 30, 35, 40]
DEFAULT_MAX_POSITIONS_OPTIONS = [2, 3, 4, 5, 8]

DEFAULT_START_DATE = "2025-01-01"
DEFAULT_FORWARD_DAYS = [1, 5, 10]
DEFAULT_SCORE_MODE = "blended"

# Extra calendar-day buffer (beyond a trading-day -> calendar-day estimate)
# so the longest Draco timeframe (5Y = 1260 trading days) is always fully
# warmed up before the requested scoring window begins.
LOOKBACK_WARMUP_CALENDAR_BUFFER = 220

SCORE_MODE_DESCRIPTIONS: dict[str, str] = {
    "precision": "Prioritizes hit rate and average return after activation.",
    "coverage": "Rewards stocks that activate often without giving up too much quality.",
    "shadow": "Scores the stock like a standalone trade-only-when-active strategy.",
    "blended": "Recommended. Mixes activation quality, activation frequency, return lift, and shadow performance.",
}


# ---------------------------------------------------------------------------
# DracoConfig (bots/draco_config.py, trimmed to what fitness scoring needs -
# DB loading dropped, defaults and validation kept)
# ---------------------------------------------------------------------------

@dataclass
class DracoConfig:
    universe: list[str]
    market_regime_ticker: str = DEFAULT_MARKET_REGIME_TICKER
    fallback_ticker: str = DEFAULT_FALLBACK_TICKER

    max_positions: int = 8
    max_new_positions_per_scan: int = 8
    scan_cadence: str = "weekly"

    entry_lookbacks: dict[str, int] = field(default_factory=lambda: dict(dm.LOOKBACKS))
    target_lookbacks: tuple[str, ...] = dm.TARGET_LOOKBACKS
    timeframe_weights: dict[str, float] = field(default_factory=lambda: dict(dm.TIMEFRAME_WEIGHTS))
    long_term_lookbacks: tuple[str, ...] = dm.LONG_TERM_LOOKBACKS

    minimum_entry_score: float = dm.DEFAULT_ENTRY_REQUIREMENTS.minimum_score
    minimum_valid_timeframes: int = dm.DEFAULT_ENTRY_REQUIREMENTS.minimum_valid_timeframes
    minimum_below_regression_timeframes: int = dm.DEFAULT_ENTRY_REQUIREMENTS.minimum_below_line_timeframes
    minimum_positive_slope_timeframes: int = dm.DEFAULT_ENTRY_REQUIREMENTS.minimum_positive_slope_timeframes
    require_positive_long_term_slope: bool = dm.DEFAULT_ENTRY_REQUIREMENTS.require_positive_long_term_slope

    minimum_target_r_squared: float = dm.DEFAULT_TARGET_REQUIREMENTS.minimum_r_squared
    require_positive_target_slope: bool = dm.DEFAULT_TARGET_REQUIREMENTS.require_positive_slope
    require_price_below_target_line: bool = dm.DEFAULT_TARGET_REQUIREMENTS.require_price_below_target_line
    minimum_entry_target_upside: float = dm.DEFAULT_TARGET_REQUIREMENTS.minimum_entry_upside
    maximum_entry_target_upside: float = dm.DEFAULT_TARGET_REQUIREMENTS.maximum_entry_upside

    stop_loss_percent: float = DEFAULT_STOP_LOSS_PERCENT
    maximum_holding_trading_days: int = DEFAULT_MAXIMUM_HOLDING_TRADING_DAYS
    symbol_reentry_cooldown_days: int = DEFAULT_SYMBOL_REENTRY_COOLDOWN_DAYS

    use_market_regime_filter: bool = True
    market_sma_period: int = DEFAULT_MARKET_SMA_PERIOD
    regime_confirmation_days: int = DEFAULT_REGIME_CONFIRMATION_DAYS
    liquidate_to_fallback_during_risk_off: bool = True

    def entry_requirements(self) -> dm.EntryRequirements:
        return dm.EntryRequirements(
            minimum_score=self.minimum_entry_score,
            minimum_valid_timeframes=self.minimum_valid_timeframes,
            minimum_below_line_timeframes=self.minimum_below_regression_timeframes,
            minimum_positive_slope_timeframes=self.minimum_positive_slope_timeframes,
            require_positive_long_term_slope=self.require_positive_long_term_slope,
            long_term_lookbacks=tuple(self.long_term_lookbacks),
        )

    def target_requirements(self) -> dm.TargetRequirements:
        return dm.TargetRequirements(
            minimum_r_squared=self.minimum_target_r_squared,
            require_positive_slope=self.require_positive_target_slope,
            require_price_below_target_line=self.require_price_below_target_line,
            minimum_entry_upside=self.minimum_entry_target_upside,
            maximum_entry_upside=self.maximum_entry_target_upside,
            candidate_labels=tuple(self.target_lookbacks),
        )


# ---------------------------------------------------------------------------
# Candidate evaluation / exit logic (bots/draco.py, ported)
# ---------------------------------------------------------------------------

@dataclass
class DracoCandidate:
    ticker: str
    total_score: float
    metrics: dict[str, dm.TimeframeMetrics]
    target: dm.LockedTarget


def evaluate_candidate(ticker: str, prices: pd.Series, cfg: DracoConfig) -> DracoCandidate | None:
    metrics = dm.compute_all_timeframe_metrics(prices.values.tolist(), lookbacks=cfg.entry_lookbacks)
    if not metrics:
        return None
    current_price = float(prices.iloc[-1])
    total_score = dm.compute_total_score(metrics, weights=cfg.timeframe_weights)

    if not dm.evaluate_entry_requirements(metrics, total_score, current_price, cfg.entry_requirements()):
        return None

    target = dm.select_locked_target(metrics, current_price, cfg.target_requirements())
    if target is None:
        return None

    return DracoCandidate(ticker=ticker, total_score=total_score, metrics=metrics, target=target)


def rank_candidates(candidates: list[DracoCandidate]) -> list[DracoCandidate]:
    return sorted(
        candidates,
        key=lambda c: (c.total_score, c.target.r_squared, c.target.entry_upside),
        reverse=True,
    )


def empty_state() -> dict[str, Any]:
    return {
        "market_regime": {"state": "risk_on", "confirmation_count": 0, "pending_state": None},
        "last_entry_scan_iso_week": None,
        "last_processed_trading_date": None,
    }


def is_new_trading_date(state: dict[str, Any], trading_date: str) -> bool:
    return state.get("last_processed_trading_date") != trading_date


def update_market_regime(
    state: dict[str, Any],
    regime_series: pd.Series,
    cfg: DracoConfig,
    *,
    is_new_trading_date_: bool,
) -> str:
    """Update (in place) and return the confirmed regime: 'risk_on' or 'risk_off'."""
    regime = state["market_regime"]
    if not cfg.use_market_regime_filter or len(regime_series) < cfg.market_sma_period:
        regime["state"] = "risk_on"
        return "risk_on"

    sma = float(regime_series.tail(cfg.market_sma_period).mean())
    current = float(regime_series.iloc[-1])
    raw_signal = "risk_off" if current < sma else "risk_on"

    if not is_new_trading_date_:
        return regime.get("state", "risk_on")

    if raw_signal == regime.get("state", "risk_on"):
        regime["pending_state"] = None
        regime["confirmation_count"] = 0
    else:
        if regime.get("pending_state") == raw_signal:
            regime["confirmation_count"] = int(regime.get("confirmation_count", 0)) + 1
        else:
            regime["pending_state"] = raw_signal
            regime["confirmation_count"] = 1

        if regime["confirmation_count"] >= cfg.regime_confirmation_days:
            regime["state"] = raw_signal
            regime["pending_state"] = None
            regime["confirmation_count"] = 0

    return regime.get("state", "risk_on")


def is_entry_scan_eligible(
    cfg: DracoConfig,
    state: dict[str, Any],
    *,
    new_trading_date: bool,
    current_iso_week: list[int],
) -> bool:
    """Weekly scans only evaluate new entries on the first eligible completed
    session of each ISO week; other cadences scan every eligible run."""
    if not new_trading_date:
        return False
    if cfg.scan_cadence != "weekly":
        return True
    return state.get("last_entry_scan_iso_week") != current_iso_week


def evaluate_position_exit(
    current_price: float,
    entry_price: float,
    holding_days: int,
    target: dm.LockedTarget,
    cfg: DracoConfig,
) -> tuple[str | None, float, float]:
    """Return (exit_reason, position_return, projected_target_price).

    exit_reason is None while the position should remain open. Priority
    order when multiple conditions are met simultaneously: stop loss, then
    locked target reached, then maximum holding period.
    """
    projected_price = dm.project_locked_target(target, holding_days)
    position_return = (current_price / entry_price) - 1.0 if entry_price > 0 else 0.0

    exit_reason: str | None = None
    if position_return <= -cfg.stop_loss_percent:
        exit_reason = "stop_loss"
    elif current_price >= projected_price:
        exit_reason = "target_reached"
    elif holding_days >= cfg.maximum_holding_trading_days:
        exit_reason = "max_hold"

    return exit_reason, position_return, projected_price


# ---------------------------------------------------------------------------
# Walk-forward grid-search fitness evaluation
# (scripts/analyze_draco_stock_fitness.py, ported)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DracoFitnessRequest:
    ticker: str
    universe: list[str]
    market_regime_ticker: str = DEFAULT_MARKET_REGIME_TICKER
    fallback_ticker: str = DEFAULT_FALLBACK_TICKER
    min_entry_scores: list[int] = field(default_factory=lambda: list(DEFAULT_MIN_ENTRY_SCORES))
    max_positions_options: list[int] = field(default_factory=lambda: list(DEFAULT_MAX_POSITIONS_OPTIONS))
    start_date: str = DEFAULT_START_DATE
    end_date: str | None = None
    forward_days: list[int] = field(default_factory=lambda: list(DEFAULT_FORWARD_DAYS))
    score_mode: str = DEFAULT_SCORE_MODE
    use_market_regime_filter: bool = True
    stop_loss_percent: float = DEFAULT_STOP_LOSS_PERCENT
    maximum_holding_trading_days: int = DEFAULT_MAXIMUM_HOLDING_TRADING_DAYS
    symbol_reentry_cooldown_days: int = DEFAULT_SYMBOL_REENTRY_COOLDOWN_DAYS


def compute_cost_drag(turnover: float, transaction_cost_bps: float = 5.0, slippage_bps: float = 5.0) -> float:
    return turnover * ((transaction_cost_bps + slippage_bps) / 10_000.0)


def _build_draco_config(request: DracoFitnessRequest, min_entry_score: int, max_positions: int) -> DracoConfig:
    return DracoConfig(
        universe=list(request.universe),
        market_regime_ticker=request.market_regime_ticker,
        fallback_ticker=request.fallback_ticker,
        max_positions=max_positions,
        max_new_positions_per_scan=max_positions,
        scan_cadence="weekly",
        minimum_entry_score=float(min_entry_score),
        use_market_regime_filter=request.use_market_regime_filter,
        stop_loss_percent=request.stop_loss_percent,
        maximum_holding_trading_days=request.maximum_holding_trading_days,
        symbol_reentry_cooldown_days=request.symbol_reentry_cooldown_days,
        liquidate_to_fallback_during_risk_off=True,
    )


def _download_draco_fitness_prices(request: DracoFitnessRequest) -> pd.DataFrame:
    """Download far enough back that every Draco timeframe (up to 5Y) is
    warmed up by the time the requested scoring window (start_date) begins."""
    longest_lookback = max(dm.LOOKBACKS.values())
    calendar_buffer_days = int(longest_lookback * 1.6) + LOOKBACK_WARMUP_CALENDAR_BUFFER
    warmup_start = (pd.Timestamp(request.start_date) - pd.Timedelta(days=calendar_buffer_days)).date().isoformat()
    end_date = request.end_date or pd.Timestamp.today().date().isoformat()
    symbols = list(dict.fromkeys([*request.universe, request.market_regime_ticker]))
    return download_adjusted_close(symbols, warmup_start, end_date)


def _analyze_combo(
    prices: pd.DataFrame,
    request: DracoFitnessRequest,
    start_idx: int,
    min_entry_score: int,
    max_positions: int,
) -> dict[str, Any]:
    cfg = _build_draco_config(request, min_entry_score, max_positions)
    universe = [t for t in cfg.universe if t in prices.columns]

    state = empty_state()
    positions: dict[str, dict[str, Any]] = {}
    cooldowns: dict[str, str] = {}

    forward_returns: dict[int, list[float]] = {horizon: [] for horizon in request.forward_days}
    unconditional_returns: dict[int, list[float]] = {horizon: [] for horizon in request.forward_days}
    activation_dates: list[str] = []
    activation_count = 0
    activation_episode_count = 0
    eligible_days = 0
    shadow_equity = 1.0
    shadow_curve = [shadow_equity]
    prev_active = False

    for idx in range(start_idx, len(prices.index)):
        trading_date_ts = prices.index[idx]
        trading_date = trading_date_ts.date().isoformat()
        new_trading_date = is_new_trading_date(state, trading_date)
        current_iso_week = list(trading_date_ts.isocalendar()[:2])
        eligible_days += 1

        for horizon in request.forward_days:
            baseline_ret = forward_return(prices, idx, request.ticker, horizon)
            if baseline_ret is not None:
                unconditional_returns[horizon].append(baseline_ret)

        if cfg.market_regime_ticker in prices.columns:
            regime_series = prices[cfg.market_regime_ticker].iloc[: idx + 1].dropna()
        else:
            regime_series = pd.Series(dtype=float)
        regime = update_market_regime(state, regime_series, cfg, is_new_trading_date_=new_trading_date)
        risk_off = regime == "risk_off"

        if risk_off and cfg.liquidate_to_fallback_during_risk_off:
            for held_ticker in list(positions.keys()):
                del positions[held_ticker]
                if cfg.symbol_reentry_cooldown_days > 0:
                    cooldown_until = trading_date_ts + pd.Timedelta(days=cfg.symbol_reentry_cooldown_days)
                    cooldowns[held_ticker] = cooldown_until.date().isoformat()
        else:
            for held_ticker in list(positions.keys()):
                if held_ticker not in prices.columns:
                    continue
                price_val = prices[held_ticker].iloc[idx]
                if pd.isna(price_val):
                    continue
                current_price = float(price_val)
                pos = positions[held_ticker]
                holding_days = idx - int(pos["entry_idx"])
                exit_reason, _position_return, _projected = evaluate_position_exit(
                    current_price, float(pos["entry_price"]), holding_days, pos["target"], cfg
                )
                if exit_reason:
                    del positions[held_ticker]
                    if cfg.symbol_reentry_cooldown_days > 0:
                        cooldown_until = trading_date_ts + pd.Timedelta(days=cfg.symbol_reentry_cooldown_days)
                        cooldowns[held_ticker] = cooldown_until.date().isoformat()

        if new_trading_date:
            cooldowns = {t: d for t, d in cooldowns.items() if d > trading_date}

        is_entry_scan_day = is_entry_scan_eligible(
            cfg, state, new_trading_date=new_trading_date, current_iso_week=current_iso_week
        )

        if is_entry_scan_day and not risk_off and len(positions) < cfg.max_positions:
            candidates: list[DracoCandidate] = []
            for candidate_ticker in universe:
                if candidate_ticker in positions or candidate_ticker in cooldowns:
                    continue
                series = prices[candidate_ticker].iloc[: idx + 1].dropna()
                if series.empty:
                    continue
                candidate = evaluate_candidate(candidate_ticker, series, cfg)
                if candidate is not None:
                    candidates.append(candidate)
            ranked = rank_candidates(candidates)
            free_slots = cfg.max_positions - len(positions)
            take = min(free_slots, cfg.max_new_positions_per_scan, len(ranked))
            for candidate in ranked[:take]:
                positions[candidate.ticker] = {
                    "entry_idx": idx,
                    "entry_price": candidate.target.entry_price,
                    "target": candidate.target,
                    "score_at_entry": candidate.total_score,
                }

        active = request.ticker in positions
        if active:
            activation_count += 1
            if not prev_active:
                activation_episode_count += 1
            activation_dates.append(trading_date)
            for horizon in request.forward_days:
                activated_ret = forward_return(prices, idx, request.ticker, horizon)
                if activated_ret is not None:
                    forward_returns[horizon].append(activated_ret)

        day_ret = forward_return(prices, idx, request.ticker, 1)
        entered_today = 0.5 if (active and not prev_active) or ((not active) and prev_active) else 0.0
        net_ret = (
            (day_ret or 0.0) - compute_cost_drag(entered_today)
            if active and day_ret is not None
            else -compute_cost_drag(entered_today)
        )
        shadow_equity *= 1.0 + net_ret
        shadow_curve.append(shadow_equity)
        prev_active = active

        if new_trading_date:
            state["last_processed_trading_date"] = trading_date
            if is_entry_scan_day:
                state["last_entry_scan_iso_week"] = current_iso_week

    avg_forward_returns = {h: (float(np.mean(v)) if v else 0.0) for h, v in forward_returns.items()}
    hit_rates = {h: (float(np.mean([value > 0.0 for value in v])) if v else 0.0) for h, v in forward_returns.items()}
    unconditional_avg = {h: (float(np.mean(v)) if v else 0.0) for h, v in unconditional_returns.items()}
    unconditional_hit = {h: (float(np.mean([value > 0.0 for value in v])) if v else 0.0) for h, v in unconditional_returns.items()}
    activation_rate = (activation_count / eligible_days) if eligible_days > 0 else 0.0
    horizon_observation_counts = {str(h): len(v) for h, v in forward_returns.items()}
    horizon_coverage = {h: (min(1.0, len(v) / float(activation_count)) if activation_count > 0 else 0.0) for h, v in forward_returns.items()}
    sample_confidence = calculate_combo_sample_confidence(
        activation_count, eligible_days, horizon_coverage, activation_episode_count=activation_episode_count,
    )
    shadow_returns = pd.Series(shadow_curve, dtype=float).pct_change().dropna()
    shadow_vol = float(shadow_returns.std()) if not shadow_returns.empty else 0.0
    shadow_sharpe = float((shadow_returns.mean() / shadow_vol) * np.sqrt(252.0)) if shadow_vol > 0.0 else 0.0

    metrics: dict[str, Any] = {
        "ticker": request.ticker,
        "lookbackDays": int(min_entry_score),
        "topN": int(max_positions),
        "minimumEntryScore": int(min_entry_score),
        "maxPositions": int(max_positions),
        "eligibleDays": int(eligible_days),
        "activationCount": int(activation_count),
        "activationEpisodeCount": int(activation_episode_count),
        "activationRate": float(activation_rate),
        "sampleConfidence": float(sample_confidence),
        "avgForwardReturns": {str(h): float(avg_forward_returns[h]) for h in request.forward_days},
        "hitRates": {str(h): float(hit_rates[h]) for h in request.forward_days},
        "baselineAvgForwardReturns": {str(h): float(unconditional_avg[h]) for h in request.forward_days},
        "baselineHitRates": {str(h): float(unconditional_hit[h]) for h in request.forward_days},
        "horizonObservationCounts": horizon_observation_counts,
        "avgForwardReturn1D": float(avg_forward_returns.get(1, 0.0)),
        "hitRate1D": float(hit_rates.get(1, 0.0)),
        "returnLift1D": float(avg_forward_returns.get(1, 0.0) - unconditional_avg.get(1, 0.0)),
        "hitRateLift1D": float(hit_rates.get(1, 0.0) - unconditional_hit.get(1, 0.0)),
        "shadowTotalReturn": float(shadow_equity - 1.0),
        "shadowMaxDrawdown": float(max_drawdown(shadow_curve)),
        "shadowSharpeLike": float(shadow_sharpe),
        "activationDatesSample": activation_dates[:10],
    }
    metrics["fitnessScore"] = float(score_combo(metrics, request.score_mode))
    metrics["comboQuality"] = float(normalize_combo_quality(metrics["fitnessScore"]))
    return metrics


def _summarize_by_entry_score(combo_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_entry_score: dict[int, list[dict[str, Any]]] = {}
    for row in combo_rows:
        by_entry_score.setdefault(int(row["minimumEntryScore"]), []).append(row)
    summary: list[dict[str, Any]] = []
    for min_entry_score, rows in sorted(by_entry_score.items()):
        best = max(rows, key=lambda item: float(item["fitnessScore"]))
        summary.append({
            "minimumEntryScore": int(min_entry_score),
            "bestMaxPositions": int(best["maxPositions"]),
            "fitnessScore": float(best["fitnessScore"]),
            "comboQuality": float(best["comboQuality"]),
            "activationRate": float(best["activationRate"]),
            "activationCount": int(best["activationCount"]),
            "sampleConfidence": float(best["sampleConfidence"]),
            "avgForwardReturn1D": float(best["avgForwardReturn1D"]),
            "returnLift1D": float(best["returnLift1D"]),
            "shadowTotalReturn": float(best["shadowTotalReturn"]),
        })
    return summary


def analyze_draco_fitness(request: DracoFitnessRequest) -> dict[str, Any]:
    """Walk-forward grid-search fitness analysis for one ticker under the
    Draco strategy family. Downloads yfinance history itself - no database,
    no API keys."""
    prices = _download_draco_fitness_prices(request)
    if prices.empty or request.ticker not in prices.columns:
        raise ValueError(f"No usable price history found for {request.ticker}.")
    prices = prices.astype("float64", copy=False)

    longest_lookback = max(dm.LOOKBACKS.values())
    if len(prices.index) <= longest_lookback:
        raise ValueError("Not enough price history to warm up Draco's longest configured (5Y) lookback window.")

    requested_start_idx = int(prices.index.searchsorted(pd.Timestamp(request.start_date)))
    start_idx = max(longest_lookback, requested_start_idx)
    if start_idx >= len(prices.index):
        raise ValueError("No trading days available in the requested date range after warmup.")

    # Walk-forward split, applied only to the scored window (start_idx
    # onward - the [0, start_idx) prefix is pure indicator warmup, never
    # scored). The grid search below only ever sees train_prices (a
    # physically truncated frame). The winning combo is then re-evaluated
    # starting at split_idx on the FULL, untruncated `prices` - Draco's
    # candidate scoring uses arbitrarily long history (up to 5Y) via
    # prices[ticker].iloc[:idx+1], so the test run needs every prior row
    # available, not just a small buffer.
    scoring_length = len(prices.index) - start_idx
    split_offset = compute_walk_forward_split_index(scoring_length, WALK_FORWARD_TRAIN_FRACTION)
    split_idx = start_idx + split_offset
    train_prices = prices.iloc[:split_idx]

    combo_rows: list[dict[str, Any]] = []
    for min_entry_score in request.min_entry_scores:
        for max_positions in request.max_positions_options:
            combo_rows.append(_analyze_combo(train_prices, request, start_idx, min_entry_score, max_positions))
    if not combo_rows:
        raise ValueError("No valid minimum-entry-score/max-positions combinations were available.")

    ranked = sorted(combo_rows, key=lambda item: float(item["fitnessScore"]), reverse=True)
    best = ranked[0]

    best_min_entry_score = int(best["minimumEntryScore"])
    best_max_positions = int(best["maxPositions"])
    out_of_sample_metrics = _analyze_combo(prices, request, split_idx, best_min_entry_score, best_max_positions)

    algorithm_fit = calculate_algorithm_fit(ranked, out_of_sample_profile=out_of_sample_metrics)
    algorithm_fit["bestProfile"] = {
        "minimumEntryScore": int(algorithm_fit["bestProfile"]["lookbackDays"]),
        "maxPositions": int(algorithm_fit["bestProfile"]["topN"]),
    }
    algorithm_fit["recommendedRegion"] = {
        "minimumEntryScores": list(algorithm_fit["recommendedRegion"]["lookbackDays"]),
        "maxPositionsOptions": list(algorithm_fit["recommendedRegion"]["topNs"]),
    }

    return {
        "algorithm": "draco",
        "ticker": request.ticker,
        "fitScope": FIT_SCOPE,
        "universeIdentity": build_universe_identity(request.universe),
        "scoreMode": request.score_mode,
        "scoreModeDescription": SCORE_MODE_DESCRIPTIONS[request.score_mode],
        "dateRange": {"start": request.start_date, "end": request.end_date},
        "config": {
            "universe": request.universe,
            "marketRegimeTicker": request.market_regime_ticker,
            "fallbackTicker": request.fallback_ticker,
            "minEntryScores": request.min_entry_scores,
            "maxPositionsOptions": request.max_positions_options,
            "useMarketRegimeFilter": request.use_market_regime_filter,
            "stopLossPercent": request.stop_loss_percent,
            "maximumHoldingTradingDays": request.maximum_holding_trading_days,
            "symbolReentryCooldownDays": request.symbol_reentry_cooldown_days,
            "forwardDays": request.forward_days,
        },
        "algorithmFit": {key: value for key, value in algorithm_fit.items() if key != "calibration"},
        "calibration": dict(algorithm_fit["calibration"]),
        "walkForward": {
            "trainFraction": WALK_FORWARD_TRAIN_FRACTION,
            "trainRows": int(split_idx - start_idx),
            "testRows": int(len(prices.index) - split_idx),
            "trainDateRange": date_bounds(prices, start_idx, split_idx - 1),
            "testDateRange": date_bounds(prices, split_idx, len(prices.index) - 1),
            "selectedProfile": {"minimumEntryScore": best_min_entry_score, "maxPositions": best_max_positions},
            "outOfSampleMetrics": out_of_sample_metrics,
        },
        "bestCombo": best,
        "entryScoreSummary": _summarize_by_entry_score(combo_rows),
        "rankedCombos": ranked,
    }


# ---------------------------------------------------------------------------
# CLI: python -m universeframe.fitness.draco --ticker NVDA
# ---------------------------------------------------------------------------

def _parse_csv_ints(raw: str | None, fallback: list[int]) -> list[int]:
    if raw is None or not str(raw).strip():
        return list(fallback)
    values: list[int] = []
    for item in str(raw).split(","):
        try:
            parsed = int(float(item.strip()))
        except ValueError:
            continue
        if parsed > 0 and parsed not in values:
            values.append(parsed)
    return values or list(fallback)


def _parse_csv_tickers(raw: str | None) -> list[str]:
    if raw is None or not str(raw).strip():
        return []
    values: list[str] = []
    for item in str(raw).split(","):
        ticker = clean_ticker(item)
        if ticker and ticker not in values:
            values.append(ticker)
    return values


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Walk-forward Draco fitness scoring for one ticker.")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--universe", help="Comma-separated universe tickers. Defaults to a sector-ETF universe; the target ticker is added automatically.")
    parser.add_argument("--market-regime-ticker", default=DEFAULT_MARKET_REGIME_TICKER)
    parser.add_argument("--fallback-ticker", default=DEFAULT_FALLBACK_TICKER)
    parser.add_argument("--min-entry-scores", default="20,25,30,35,40")
    parser.add_argument("--max-positions", default="2,3,4,5,8")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--forward-days", default="1,5,10")
    parser.add_argument("--score-mode", default=DEFAULT_SCORE_MODE, choices=sorted(SCORE_MODE_DESCRIPTIONS))
    parser.add_argument("--use-market-regime-filter", dest="use_market_regime_filter", action="store_true")
    parser.add_argument("--no-market-regime-filter", dest="use_market_regime_filter", action="store_false")
    parser.set_defaults(use_market_regime_filter=True)
    parser.add_argument("--stop-loss-percent", type=float, default=DEFAULT_STOP_LOSS_PERCENT)
    parser.add_argument("--maximum-holding-trading-days", type=int, default=DEFAULT_MAXIMUM_HOLDING_TRADING_DAYS)
    parser.add_argument("--symbol-reentry-cooldown-days", type=int, default=DEFAULT_SYMBOL_REENTRY_COOLDOWN_DAYS)
    return parser


def _build_request(args: argparse.Namespace) -> DracoFitnessRequest:
    ticker = clean_ticker(args.ticker)
    universe = _parse_csv_tickers(args.universe) or list(DEFAULT_DRACO_UNIVERSE)
    if ticker and ticker not in universe:
        universe.append(ticker)
    return DracoFitnessRequest(
        ticker=ticker,
        universe=universe,
        market_regime_ticker=clean_ticker(args.market_regime_ticker, DEFAULT_MARKET_REGIME_TICKER),
        fallback_ticker=clean_ticker(args.fallback_ticker, DEFAULT_FALLBACK_TICKER),
        min_entry_scores=_parse_csv_ints(args.min_entry_scores, DEFAULT_MIN_ENTRY_SCORES),
        max_positions_options=_parse_csv_ints(args.max_positions, DEFAULT_MAX_POSITIONS_OPTIONS),
        start_date=args.start_date,
        end_date=args.end_date,
        forward_days=_parse_csv_ints(args.forward_days, DEFAULT_FORWARD_DAYS),
        score_mode=args.score_mode,
        use_market_regime_filter=bool(args.use_market_regime_filter),
        stop_loss_percent=float(args.stop_loss_percent),
        maximum_holding_trading_days=int(args.maximum_holding_trading_days),
        symbol_reentry_cooldown_days=int(args.symbol_reentry_cooldown_days),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        result = analyze_draco_fitness(_build_request(args))
    except Exception as exc:
        json.dump({"success": False, "error": str(exc)}, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 1
    json.dump({"success": True, **result}, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
