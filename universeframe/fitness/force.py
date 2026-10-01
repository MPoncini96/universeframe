"""Force fitness scoring - a faithful, standalone port of Monstra's
``bots/alpha1.py`` holdings-selection logic plus
``scripts/analyze_force_stock_fitness.py``'s walk-forward grid search.

Force picks a fixed number of top trailing-return names out of a universe
each day, weights them by rank (e.g. 40/30/20/10), and defends both
membership and rank position against noisy lead changes (the "incumbency
margin"). Fitness asks: if a given ticker had been sitting in Force's
universe this whole time, how good a Force holding would it actually have
been, across a grid of (lookback days, portfolio size) combos - validated
out-of-sample, not just on whichever combo happened to look best in-sample?
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from functools import cmp_to_key
from typing import Any

import numpy as np
import pandas as pd

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
    slice_test_prices_with_lookback_buffer,
    FIT_SCOPE,
    WALK_FORWARD_TRAIN_FRACTION,
)
from .prices import download_adjusted_close

DEFAULT_TOP_N = 4
DEFAULT_WEIGHTS = np.array([0.40, 0.30, 0.20, 0.10], dtype=float)

# Only act on a lookback window with no missing data inside it - a partial
# window would silently understate a trailing return for any ticker with a
# gap, which would bias the ranking that gap-free tickers compete against.
REQUIRE_FULL_LOOKBACK_WINDOW = True
USE_CASH_EQUIVALENT_FALLBACK = True

DEFAULT_TRANSACTION_COST_BPS = 5.0
DEFAULT_SLIPPAGE_BPS = 5.0

# Turnover buffer: a currently-held name (and its rank position, since
# positions carry fixed weights like 40/30/20/10) is only displaced or
# reordered if a rival's trailing return clears its own return by more than
# this fraction of its own return's magnitude. Prevents both membership
# churn (swapping which stocks are held) and weight-only churn (the same
# holdings reshuffling rank position on noise).
DEFAULT_INCUMBENCY_MARGIN = 0.15

DEFAULT_FORCE_UNIVERSE = [
    "XLK", "XLF", "XLV", "XLY", "XLI", "XLP", "XLE", "XLB", "XLU", "XLRE", "XLC", "SMH",
]

# Adaptive-universe per-lookback labels: these exact lookback-day values must
# be present in the tested grid for the corresponding force_10d/force_15d/
# force_45d/force_3m adaptive-universe variants to each get a distinct, real
# score. "3m" maps to 63 trading days (~3 calendar months) rather than a
# calendar count, matching the production grid's longest tested lookback.
LOOKBACK_LABELS: dict[int, str] = {10: "10d", 15: "15d", 45: "45d", 63: "3m"}

DEFAULT_START_DATE = "2025-01-01"
DEFAULT_FORWARD_DAYS = [1, 5, 10]
DEFAULT_SCORE_MODE = "blended"

SCORE_MODE_DESCRIPTIONS: dict[str, str] = {
    "precision": "Prioritizes hit rate and average return after activation.",
    "coverage": "Rewards stocks that activate often without giving up too much quality.",
    "shadow": "Scores the stock like a standalone trade-only-when-active strategy.",
    "blended": "Recommended. Mixes activation quality, activation frequency, return lift, and shadow performance.",
}


# ---------------------------------------------------------------------------
# Force's live holdings-selection logic (bots/alpha1.py, ported verbatim)
# ---------------------------------------------------------------------------

@dataclass
class Alpha1Config:
    universe: list[str]
    cash_equivalent: str | None
    top_n: int | None
    lookback_days: int
    enable_kill_switch: bool = True
    kill_switch_mode: str = "top_negative"
    top_return_threshold: float = 0.0
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS
    incumbency_margin: float = DEFAULT_INCUMBENCY_MARGIN
    rank_weights: np.ndarray = field(default_factory=lambda: DEFAULT_WEIGHTS.copy())


def normalize_rank_weights(raw: Any) -> np.ndarray:
    """Coerce arbitrary weights input to a positive simplex; fallback to DEFAULT_WEIGHTS."""
    if raw is None:
        return DEFAULT_WEIGHTS.copy()
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return DEFAULT_WEIGHTS.copy()
    if not isinstance(raw, (list, tuple)) or len(raw) == 0:
        return DEFAULT_WEIGHTS.copy()
    try:
        arr = np.array([float(x) for x in raw], dtype=float)
    except (TypeError, ValueError):
        return DEFAULT_WEIGHTS.copy()
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        return DEFAULT_WEIGHTS.copy()
    s = float(arr.sum())
    if s > 1.5:
        arr = arr / 100.0
        s = float(arr.sum())
    if s <= 0:
        return DEFAULT_WEIGHTS.copy()
    return arr / s


def clamp_top_n(top_n: int | None, rank_weights: np.ndarray) -> int:
    cap = int(len(rank_weights)) if len(rank_weights) > 0 else DEFAULT_TOP_N
    cap = max(1, cap)
    if top_n is None:
        return min(DEFAULT_TOP_N, cap)
    return max(1, min(int(top_n), cap))


def get_trailing_returns(
    prices: pd.DataFrame,
    end_idx_exclusive: int,
    lookback_days: int,
    symbols: set[str] | None = None,
) -> pd.Series:
    start_idx = max(0, end_idx_exclusive - lookback_days)
    lookback = prices.iloc[start_idx:end_idx_exclusive]

    if len(lookback) < 2:
        return pd.Series(dtype=float)

    first_row = lookback.iloc[0]
    last_row = lookback.iloc[-1]

    if REQUIRE_FULL_LOOKBACK_WINDOW:
        valid_mask = lookback.notna().all(axis=0)
        first_row = first_row[valid_mask]
        last_row = last_row[valid_mask]
    else:
        valid_mask = first_row.notna() & last_row.notna()
        first_row = first_row[valid_mask]
        last_row = last_row[valid_mask]

    trailing = (last_row / first_row) - 1.0
    trailing = trailing.replace([np.inf, -np.inf], np.nan).dropna()

    if symbols is not None:
        trailing = trailing[trailing.index.isin(symbols)]

    return trailing


def _evaluate_kill_switch(config: Alpha1Config, ranked_trailing: pd.Series) -> tuple[bool, str]:
    if config.enable_kill_switch and config.kill_switch_mode != "off":
        if ranked_trailing.empty:
            return True, "kill_switch:no_ranked_candidates"
        if config.kill_switch_mode == "top_negative":
            top_ret = float(ranked_trailing.max())
            if top_ret <= config.top_return_threshold:
                return True, f"kill_switch:top_ret={top_ret:.6f}<=threshold={config.top_return_threshold:.6f}"
        elif config.kill_switch_mode == "avg_negative":
            avg_ret = float(ranked_trailing.mean())
            if avg_ret <= config.top_return_threshold:
                return True, f"kill_switch:avg_ret={avg_ret:.6f}<=threshold={config.top_return_threshold:.6f}"
    return False, "risk_on"


def rank_with_stability(
    ranked_trailing: pd.Series,
    top_n: int,
    current_holdings: list[str] | None,
    margin: float,
) -> list[str]:
    """Order candidates into the final top_n, defending both a currently-held
    name's membership AND its relative rank position (since position carries
    a fixed weight like 40/30/20/10) unless a rival clears its margin.

    A single comparator handles all three pairings:
      - two incumbents: whichever was senior (held the better slot) stays
        ahead unless the junior's return clears the senior's own
        margin-scaled cushion.
      - incumbent vs. challenger: the incumbent defends its slot the same
        way, which is what stops membership churn.
      - two challengers: plain return ranking, no history to defend.
    """
    if ranked_trailing.empty:
        return []
    returns = ranked_trailing.to_dict()
    prev_rank = {symbol: idx for idx, symbol in enumerate(current_holdings or []) if symbol in returns}

    def cushion(symbol: str) -> float:
        r = returns[symbol]
        return r + margin * abs(r)

    def before(a: str, b: str) -> bool:
        a_rank, b_rank = prev_rank.get(a), prev_rank.get(b)
        if a_rank is not None and b_rank is not None:
            senior, junior = (a, b) if a_rank < b_rank else (b, a)
            junior_wins = returns[junior] > cushion(senior)
            return (senior == a) != junior_wins
        if a_rank is not None:
            return not (returns[b] > cushion(a))
        if b_rank is not None:
            return returns[a] > cushion(b)
        return returns[a] > returns[b]

    def compare(a: str, b: str) -> int:
        if a == b:
            return 0
        return -1 if before(a, b) else 1

    ordered = sorted(returns.keys(), key=cmp_to_key(compare))
    return ordered[:top_n]


def choose_holdings_for_day(
    config: Alpha1Config,
    ranked_trailing: pd.Series,
    current_holdings: list[str] | None = None,
) -> tuple[list[str], np.ndarray, bool, str]:
    risk_off, reason = _evaluate_kill_switch(config, ranked_trailing)

    if risk_off:
        if USE_CASH_EQUIVALENT_FALLBACK and config.cash_equivalent:
            return [config.cash_equivalent], np.array([1.0], dtype=float), True, reason
        return [], np.array([], dtype=float), True, reason

    selected = rank_with_stability(ranked_trailing, int(config.top_n or DEFAULT_TOP_N), current_holdings, config.incumbency_margin)
    rw = np.asarray(config.rank_weights, dtype=float)
    weights = rw[: len(selected)].copy()
    if len(weights) == 0:
        selected, weights = [], np.array([], dtype=float)
    else:
        weights = weights / weights.sum()

    if not selected and USE_CASH_EQUIVALENT_FALLBACK and config.cash_equivalent:
        return [config.cash_equivalent], np.array([1.0], dtype=float), True, "fallback:no_selected_symbols"

    return selected, weights, False, reason


def compute_cost_drag(turnover: float, transaction_cost_bps: float, slippage_bps: float) -> float:
    """Total cost drag in decimal return terms, e.g. turnover=1.0,
    tc=5bps, slip=5bps -> 0.001 (10bps total drag)."""
    total_bps = float(transaction_cost_bps) + float(slippage_bps)
    return turnover * (total_bps / 10_000.0)


# ---------------------------------------------------------------------------
# Walk-forward grid-search fitness evaluation
# (scripts/analyze_force_stock_fitness.py, ported)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ForceFitnessRequest:
    ticker: str
    universe: list[str]
    safety_net_equity: str = "VOO"
    lookbacks: list[int] = field(default_factory=lambda: [10, 14, 15, 21, 30, 45, 63])
    portfolio_sizes: list[int] = field(default_factory=lambda: [2, 3, 4, 5])
    weights: list[float] = field(default_factory=lambda: [0.40, 0.30, 0.20, 0.10])
    start_date: str = DEFAULT_START_DATE
    end_date: str | None = None
    forward_days: list[int] = field(default_factory=lambda: list(DEFAULT_FORWARD_DAYS))
    score_mode: str = DEFAULT_SCORE_MODE


def _build_force_config(request: ForceFitnessRequest, lookback: int, portfolio_size: int) -> Alpha1Config:
    normalized_weights = normalize_rank_weights(request.weights)
    effective_size = min(max(1, portfolio_size), len(request.universe), len(normalized_weights))
    return Alpha1Config(
        universe=request.universe,
        cash_equivalent=request.safety_net_equity,
        top_n=clamp_top_n(effective_size, normalized_weights),
        lookback_days=int(lookback),
        enable_kill_switch=True,
        kill_switch_mode="top_negative",
        top_return_threshold=0.0,
        rank_weights=normalized_weights,
    )


def _build_price_frame(universe: list[str], cash_equivalent: str, start_date: str, end_date: str) -> pd.DataFrame:
    symbols = list(dict.fromkeys([*universe, cash_equivalent]))
    return download_adjusted_close(symbols, start_date, end_date)


def _analyze_combo(prices: pd.DataFrame, request: ForceFitnessRequest, lookback: int, portfolio_size: int) -> dict[str, Any]:
    config = _build_force_config(request, lookback, portfolio_size)
    ranking_universe = set(config.universe)
    forward_returns: dict[int, list[float]] = {horizon: [] for horizon in request.forward_days}
    unconditional_returns: dict[int, list[float]] = {horizon: [] for horizon in request.forward_days}
    activation_count = 0
    activation_episode_count = 0
    eligible_days = 0
    activation_dates: list[str] = []
    shadow_equity = 1.0
    shadow_curve = [shadow_equity]
    prev_active = False
    current_holdings: list[str] | None = None

    for decision_idx in range(1, len(prices.index)):
        if decision_idx < lookback:
            continue
        ranked_trailing = get_trailing_returns(prices, decision_idx, lookback, ranking_universe)
        selected_symbols, _weights, risk_off, _risk_reason = choose_holdings_for_day(
            config, ranked_trailing, current_holdings=current_holdings,
        )
        if not risk_off:
            current_holdings = list(selected_symbols)
        if ranked_trailing.empty:
            continue
        eligible_days += 1
        for horizon in request.forward_days:
            unconditional_ret = forward_return(prices, decision_idx, request.ticker, horizon)
            if unconditional_ret is not None:
                unconditional_returns[horizon].append(unconditional_ret)
        active = (not risk_off) and (request.ticker in selected_symbols)
        if active:
            activation_count += 1
            if not prev_active:
                activation_episode_count += 1
            activation_dates.append(str(prices.index[decision_idx].date()))
            for horizon in request.forward_days:
                activated_ret = forward_return(prices, decision_idx, request.ticker, horizon)
                if activated_ret is not None:
                    forward_returns[horizon].append(activated_ret)
        day_ret = forward_return(prices, decision_idx, request.ticker, 1)
        entered_today = 0.5 if (active and not prev_active) or ((not active) and prev_active) else 0.0
        drag = compute_cost_drag(entered_today, config.transaction_cost_bps, config.slippage_bps)
        net_ret = ((day_ret or 0.0) - drag) if active and day_ret is not None else -drag
        shadow_equity *= 1.0 + net_ret
        shadow_curve.append(shadow_equity)
        prev_active = active

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
        "lookbackDays": int(lookback),
        "topN": int(portfolio_size),
        "portfolioSize": int(portfolio_size),
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


def _summarize_by_lookback(combo_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_lookback: dict[int, list[dict[str, Any]]] = {}
    for row in combo_rows:
        by_lookback.setdefault(int(row["lookbackDays"]), []).append(row)
    summary: list[dict[str, Any]] = []
    for lookback, rows in sorted(by_lookback.items()):
        best = max(rows, key=lambda item: float(item["fitnessScore"]))
        summary.append({
            "lookbackDays": int(lookback),
            "bestPortfolioSize": int(best["portfolioSize"]),
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


def _lookback_quality_by_label_out_of_sample(
    combo_rows: list[dict[str, Any]], prices: pd.DataFrame, split_idx: int, request: ForceFitnessRequest,
) -> dict[str, float]:
    """Compact {label: comboQuality} map for the adaptive-universe
    force_10d/force_15d/force_45d/force_3m variants: re-evaluates each
    labeled lookback's train-selected portfolio size against the held-out
    test slice, the same out-of-sample validation the overall best combo
    gets, applied per label instead of just once overall."""
    by_lookback: dict[int, list[dict[str, Any]]] = {}
    for row in combo_rows:
        by_lookback.setdefault(int(row["lookbackDays"]), []).append(row)

    quality_by_label: dict[str, float] = {}
    for lookback, rows in by_lookback.items():
        label = LOOKBACK_LABELS.get(lookback)
        if label is None:
            continue
        best = max(rows, key=lambda item: float(item["fitnessScore"]))
        test_prices = slice_test_prices_with_lookback_buffer(prices, split_idx, lookback)
        oos_metrics = _analyze_combo(test_prices, request, lookback, int(best["portfolioSize"]))
        quality_by_label[label] = float(normalize_combo_quality(oos_metrics["fitnessScore"]))
    return quality_by_label


def analyze_force_fitness(request: ForceFitnessRequest) -> dict[str, Any]:
    """Walk-forward grid-search fitness analysis for one ticker under the
    Force strategy family. Downloads yfinance history itself - no database,
    no API keys."""
    end_date = request.end_date or pd.Timestamp.today().date().isoformat()
    prices = _build_price_frame(request.universe, request.safety_net_equity, request.start_date, end_date)
    if prices.empty or request.ticker not in prices.columns:
        raise ValueError(f"No usable price history found for {request.ticker}.")
    prices = prices.astype("float64", copy=False)

    # Walk-forward split: the grid search below (which selects bestCombo)
    # only ever sees train_prices. The winning combo is then re-evaluated on
    # test_prices - data it never influenced - and that held-out result is
    # what actually feeds algorithmFit.score. This is what stops "best of a
    # grid search" multiple-comparisons bias from reporting its own
    # in-sample number as if it were unbiased evidence.
    split_idx = compute_walk_forward_split_index(len(prices.index))
    train_prices = prices.iloc[:split_idx]

    combo_rows: list[dict[str, Any]] = []
    for lookback in request.lookbacks:
        for portfolio_size in request.portfolio_sizes:
            combo_rows.append(_analyze_combo(train_prices, request, lookback, portfolio_size))
    ranked = sorted(combo_rows, key=lambda item: float(item["fitnessScore"]), reverse=True)
    best = ranked[0]

    best_lookback = int(best["lookbackDays"])
    best_portfolio_size = int(best["portfolioSize"])
    test_prices = slice_test_prices_with_lookback_buffer(prices, split_idx, best_lookback)
    out_of_sample_metrics = _analyze_combo(test_prices, request, best_lookback, best_portfolio_size)

    algorithm_fit = calculate_algorithm_fit(ranked, out_of_sample_profile=out_of_sample_metrics)
    algorithm_fit["bestProfile"] = {
        "lookbackDays": int(algorithm_fit["bestProfile"]["lookbackDays"]),
        "portfolioSize": int(algorithm_fit["bestProfile"]["topN"]),
    }
    algorithm_fit["recommendedRegion"] = {
        "lookbackDays": list(algorithm_fit["recommendedRegion"]["lookbackDays"]),
        "portfolioSizes": list(algorithm_fit["recommendedRegion"]["topNs"]),
    }
    lookback_summary = _summarize_by_lookback(combo_rows)
    lookback_quality_by_label = _lookback_quality_by_label_out_of_sample(combo_rows, prices, split_idx, request)

    return {
        "algorithm": "force",
        "ticker": request.ticker,
        "fitScope": FIT_SCOPE,
        "universeIdentity": build_universe_identity(request.universe),
        "scoreMode": request.score_mode,
        "scoreModeDescription": SCORE_MODE_DESCRIPTIONS[request.score_mode],
        "dateRange": {"start": request.start_date, "end": end_date},
        "config": {
            "universe": request.universe,
            "safetyNetEquity": request.safety_net_equity,
            "lookbacks": request.lookbacks,
            "portfolioSizes": request.portfolio_sizes,
            "weights": request.weights,
            "forwardDays": request.forward_days,
        },
        "algorithmFit": {key: value for key, value in algorithm_fit.items() if key != "calibration"},
        "calibration": dict(algorithm_fit["calibration"]),
        "walkForward": {
            "trainFraction": WALK_FORWARD_TRAIN_FRACTION,
            "trainRows": int(split_idx),
            "testRows": int(len(prices.index) - split_idx),
            "trainDateRange": date_bounds(prices, 0, split_idx - 1),
            "testDateRange": date_bounds(prices, split_idx, len(prices.index) - 1),
            "selectedProfile": {"lookbackDays": best_lookback, "portfolioSize": best_portfolio_size},
            "outOfSampleMetrics": out_of_sample_metrics,
        },
        "bestCombo": best,
        "lookbackSummary": lookback_summary,
        "lookbackQualityByLabel": lookback_quality_by_label,
        "rankedCombos": ranked,
    }


# ---------------------------------------------------------------------------
# CLI: python -m universeframe.fitness.force --ticker NVDA
# ---------------------------------------------------------------------------

def _parse_csv_ints(raw: str | None, fallback: list[int]) -> list[int]:
    if raw is None or not str(raw).strip():
        return list(fallback)
    values: list[int] = []
    for item in str(raw).split(","):
        try:
            parsed = int(item.strip())
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
    parser = argparse.ArgumentParser(description="Walk-forward Force fitness scoring for one ticker.")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--universe", help="Comma-separated universe tickers. Defaults to a sector-ETF universe; the target ticker is added automatically.")
    parser.add_argument("--safety-net-equity", default="VOO")
    parser.add_argument("--lookbacks", default="10,14,15,21,30,45,63")
    parser.add_argument("--portfolio-sizes", default="2,3,4,5")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--forward-days", default="1,5,10")
    parser.add_argument("--score-mode", default=DEFAULT_SCORE_MODE, choices=sorted(SCORE_MODE_DESCRIPTIONS))
    return parser


def _build_request(args: argparse.Namespace) -> ForceFitnessRequest:
    ticker = clean_ticker(args.ticker)
    universe = _parse_csv_tickers(args.universe) or list(DEFAULT_FORCE_UNIVERSE)
    if ticker and ticker not in universe:
        universe.append(ticker)
    return ForceFitnessRequest(
        ticker=ticker,
        universe=universe,
        safety_net_equity=clean_ticker(args.safety_net_equity, "VOO"),
        lookbacks=_parse_csv_ints(args.lookbacks, [10, 14, 15, 21, 30, 45, 63]),
        portfolio_sizes=_parse_csv_ints(args.portfolio_sizes, [2, 3, 4, 5]),
        start_date=args.start_date,
        end_date=args.end_date,
        forward_days=_parse_csv_ints(args.forward_days, DEFAULT_FORWARD_DAYS),
        score_mode=args.score_mode,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        result = analyze_force_fitness(_build_request(args))
    except Exception as exc:
        json.dump({"success": False, "error": str(exc)}, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 1
    json.dump({"success": True, **result}, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
