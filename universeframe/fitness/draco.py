"""Draco fitness scoring - a faithful, standalone port of Monstra's
``scripts/analyze_draco_stock_fitness.py``.

Draco is a multi-timeframe log-linear regression mean-reversion strategy
(see ``draco_math.py``): each week it scans a universe for candidates
trading below a strong upward trend line. Fitness asks a narrower,
ticker-only question: when this ticker has dipped notably below its own
regression trend line, how often and how far does it actually come back?
For each configured timeframe, it measures (out of sample) the empirical
probability that a dip like today's has historically closed within that
timeframe's reversion horizon, multiplies by how large today's gap actually
is, and sums across timeframes using Draco's existing per-timeframe
weights.

What's intentionally left out, relative to earlier versions of this module
and the live bot: any universe/portfolio simulation (candidate ranking,
locked-target entries/exits, stop loss, max holding period, market-regime
risk-off liquidation, a portfolio-level circuit breaker). None of that
affects whether a single ticker tends to recover from its own dips, so this
scores one ticker's own price history in isolation.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any

import pandas as pd

from . import draco_math as dm
from .common import clean_ticker, date_bounds
from .prices import download_adjusted_close

# Rolling scored window: always the last ROLLING_SCORE_WINDOW_YEARS years
# ending at end_date, rather than a fixed calendar anchor -- a fixed anchor
# would make the scored window (and this analyzer's per-day timeframe scan)
# grow longer, and slower, by one day every day it's run. Starting choice,
# not empirically tuned: long enough to observe a meaningful number of dip
# episodes per timeframe, short enough to stay cheap.
ROLLING_SCORE_WINDOW_YEARS = 2.0
_DAYS_PER_YEAR = 365.25

DEFAULT_TRAIN_FRACTION = 0.70

# Ticker-only: this ticker's own price history is scored in isolation, with
# no universe or position-slot competition (unlike Force/Aptet's
# universe_conditioned FIT_SCOPE in common.py).
FIT_SCOPE = "ticker_only"

# Extra calendar-day buffer (beyond a trading-day -> calendar-day estimate)
# so the longest Draco timeframe (5Y = 1260 trading days) is always fully
# warmed up before the requested scoring window begins.
LOOKBACK_WARMUP_CALENDAR_BUFFER = 220


@dataclass(frozen=True)
class DracoFitnessRequest:
    ticker: str
    start_date: str | None = None  # None -> rolling ROLLING_SCORE_WINDOW_YEARS years before end_date
    end_date: str | None = None  # None -> resolved to "today" when the request is run
    train_fraction: float = DEFAULT_TRAIN_FRACTION


def _resolve_end_date(end_date: str | None) -> str:
    return end_date or pd.Timestamp.today().date().isoformat()


def _resolve_start_date(start_date: str | None, end_date: str) -> str:
    if start_date:
        return start_date
    window_days = int(round(ROLLING_SCORE_WINDOW_YEARS * _DAYS_PER_YEAR))
    return (pd.Timestamp(end_date) - pd.Timedelta(days=window_days)).date().isoformat()


def _download_draco_fitness_prices(ticker: str, start_date: str, end_date: str) -> pd.Series:
    """Download far enough back that every Draco timeframe (up to 5Y) is
    warmed up by the time the requested scoring window (start_date) begins."""
    longest_lookback = max(dm.LOOKBACKS.values())
    calendar_buffer_days = int(longest_lookback * 1.6) + LOOKBACK_WARMUP_CALENDAR_BUFFER
    warmup_start = (pd.Timestamp(start_date) - pd.Timedelta(days=calendar_buffer_days)).date().isoformat()
    frame = download_adjusted_close([ticker], warmup_start, end_date)
    if frame.empty or ticker not in frame.columns:
        return pd.Series(dtype="float64")
    return frame[ticker].dropna().astype("float64")


def _profile_to_dict(profile: "dm.TimeframeReversionProfile") -> dict[str, Any]:
    return {
        "label": profile.label,
        "lookback": int(profile.lookback),
        "weight": float(profile.weight),
        "zScoreNow": float(profile.z_score_now),
        "isDippedNow": bool(profile.z_score_now < 0.0),
        "distanceToMeanNow": float(profile.distance_to_mean_now),
        "reversionProbabilityTrain": float(profile.reversion_probability_train),
        "reversionProbabilityTest": (
            float(profile.reversion_probability_test) if profile.reversion_probability_test is not None else None
        ),
        "episodeCountTrain": int(profile.episode_count_train),
        "episodeCountTest": int(profile.episode_count_test),
        "expectedValue": float(profile.expected_value),
        "weightedExpectedValue": float(profile.expected_value * profile.weight),
    }


def analyze_draco_fitness(request: DracoFitnessRequest) -> dict[str, Any]:
    """Walk-forward mean-reversion fitness analysis for one ticker under the
    Draco strategy family. Downloads yfinance history itself - no database,
    no API keys."""
    end_date = _resolve_end_date(request.end_date)
    start_date = _resolve_start_date(request.start_date, end_date)

    prices = _download_draco_fitness_prices(request.ticker, start_date, end_date)
    if prices.empty:
        raise ValueError(f"No usable price history found for {request.ticker}.")

    longest_lookback = max(dm.LOOKBACKS.values())
    if len(prices.index) <= longest_lookback:
        raise ValueError("Not enough price history to warm up Draco's longest configured (5Y) lookback window.")

    requested_start_idx = int(prices.index.searchsorted(pd.Timestamp(start_date)))

    fitness = dm.compute_mean_reversion_fitness(
        prices.to_numpy(dtype="float64"),
        request.ticker,
        start_index=requested_start_idx,
        train_fraction=request.train_fraction,
    )
    if fitness is None:
        raise ValueError("No trading days available in the requested date range after warmup.")

    return {
        "algorithm": "draco",
        "ticker": request.ticker,
        "fitScope": FIT_SCOPE,
        "dateRange": {"start": start_date, "end": end_date},
        "config": {
            "trainFraction": request.train_fraction,
            "dipZThreshold": dm.DIP_Z_THRESHOLD,
            "reversionHorizonDays": dict(dm.REVERSION_HORIZON_DAYS),
        },
        "algorithmFit": {
            "score": float(fitness.score),
            "rawScore": float(fitness.raw_score),
            "confidence": float(fitness.confidence),
            "label": fitness.label,
            "totalScore": float(fitness.total_score),
        },
        "walkForward": {
            "trainFraction": request.train_fraction,
            "trainDateRange": date_bounds(prices, fitness.start_index, fitness.split_index - 1),
            "testDateRange": date_bounds(prices, fitness.split_index, len(prices.index) - 1),
        },
        "timeframeBreakdown": [_profile_to_dict(p) for p in fitness.profiles],
        "interpretation": {
            "algorithmFitMeaning": "Measures how reliably this ticker has historically closed a meaningful gap below its own regression trend line, weighted by how large today's gap actually is -- not whether a simulated Draco portfolio would have made money trading it.",
            "fitScopeMeaning": "Draco fit is ticker-only: this ticker's own price history is scored in isolation, with no universe or position-slot competition.",
            "backendMeaning": "Reuses Draco's live multi-timeframe log-linear regression (draco_math.py) for the z-score/fitted-trend-line math, but replaces the live entry/exit/stop-loss/circuit-breaker simulation with a direct empirical reversion-probability measurement.",
            "walkForwardMeaning": "Per-timeframe reversion probabilities are measured on the train portion of the scored window (walkForward.trainDateRange) and re-measured independently on the held-out test portion (walkForward.testDateRange); the test-period probability (when there's any test-period evidence) is what feeds algorithmFit, not the in-sample one.",
            "timeframeBreakdownMeaning": "Per configured lookback: today's z-score and % distance below the fitted trend line, this ticker's historical probability of closing that kind of gap within the timeframe's reversion horizon, and the resulting expected value (probability x distance, weighted). Only timeframes where the ticker is currently below its trend line (isDippedNow) contribute a nonzero expected value.",
        },
    }


# ---------------------------------------------------------------------------
# CLI: python -m universeframe.fitness.draco --ticker NVDA
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze how good a mean-reversion candidate a stock is for Draco: the empirical probability it closes a gap below its own trend line, times how large that gap is today.")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--start-date", default=None, help=f"Defaults to a rolling {ROLLING_SCORE_WINDOW_YEARS:g} years before --end-date.")
    parser.add_argument("--end-date", default=None, help="Defaults to today.")
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION, help="Walk-forward train/test split fraction (default: 0.70).")
    return parser


def _build_request(args: argparse.Namespace) -> DracoFitnessRequest:
    return DracoFitnessRequest(
        ticker=clean_ticker(args.ticker),
        start_date=args.start_date,
        end_date=args.end_date,
        train_fraction=float(args.train_fraction),
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
