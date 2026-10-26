"""Pure regression/scoring/target functions for the Draco algorithm family.

A verbatim port of Monstra-Worker's ``draco_math.py``. No I/O; every function
here is deterministic given plain numeric inputs. Draco fits
``ln(price) = intercept + slope * index`` over each configured lookback,
scores candidates on how far below their regression line they trade, how
strong and consistent their trend is, and how well the line fits, then locks
a regression-projected exit target at entry that is never recalculated.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Constants (Draco strategy defaults)
# ---------------------------------------------------------------------------

LOOKBACKS: dict[str, int] = {
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "1Y": 252,
    "2Y": 504,
    "3Y": 756,
    "4Y": 1008,
    "5Y": 1260,
}

TIMEFRAME_WEIGHTS: dict[str, float] = {
    "1M": 0.75,
    "3M": 1.00,
    "6M": 1.15,
    "1Y": 1.30,
    "2Y": 1.50,
    "3Y": 1.70,
    "4Y": 1.85,
    "5Y": 2.00,
}

TARGET_LOOKBACKS: tuple[str, ...] = ("3M", "6M", "1Y", "2Y")
LONG_TERM_LOOKBACKS: tuple[str, ...] = ("2Y", "3Y", "4Y", "5Y")

Z_SCORE_CLAMP = 4.0
ANNUALIZED_SLOPE_EXPONENT_CLAMP = 20.0
EXP_OVERFLOW_GUARD = 700.0  # math.exp overflows ~709.78 for float64
TRADING_DAYS_PER_YEAR = 252

MINIMUM_WINDOW_LENGTH = 3


def _clamp(value: float, lo: float, hi: float) -> float:
    if not math.isfinite(value):
        return hi if value > 0 else lo
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Regression fit
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TimeframeMetrics:
    label: str
    lookback: int
    slope: float
    intercept: float
    endpoint_index: int
    current_price: float
    fitted_price: float
    residual_std: float
    z_score: float
    annualized_slope: float
    r_squared: float


def fit_log_linear_regression(
    prices: Sequence[float],
) -> tuple[float, float, np.ndarray, np.ndarray] | None:
    """OLS fit of ln(price) = intercept + slope * index over `prices`.

    Returns (slope, intercept, index_array, log_prices), or None if the series
    is too short or contains non-finite / non-positive values.
    """
    arr = np.asarray(prices, dtype=float)
    if arr.size < MINIMUM_WINDOW_LENGTH:
        return None
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
        return None

    index = np.arange(arr.size, dtype=float)
    log_prices = np.log(arr)
    slope, intercept = np.polyfit(index, log_prices, 1)
    return float(slope), float(intercept), index, log_prices


def compute_timeframe_metrics(
    label: str,
    lookback: int,
    prices: Sequence[float],
) -> TimeframeMetrics | None:
    """Compute regression metrics for one timeframe using the trailing `lookback` prices.

    `prices` should be the full adjusted-close history ending at "today"; only
    the trailing `lookback` entries are used. Returns None when there isn't
    enough valid history for this timeframe.
    """
    if len(prices) < lookback:
        return None
    window = prices[-lookback:]
    fit = fit_log_linear_regression(window)
    if fit is None:
        return None
    slope, intercept, index, log_prices = fit

    endpoint_index = len(window) - 1
    fitted_log = intercept + slope * index
    residuals = log_prices - fitted_log
    residual_std = float(np.std(residuals, ddof=1)) if len(residuals) > 2 else 0.0

    current_price = float(window[-1])
    fitted_current_log = intercept + slope * endpoint_index
    fitted_price = math.exp(_clamp(fitted_current_log, -EXP_OVERFLOW_GUARD, EXP_OVERFLOW_GUARD))

    if residual_std > 1e-12:
        raw_z = (math.log(current_price) - fitted_current_log) / residual_std
    else:
        raw_z = 0.0
    z_score = _clamp(raw_z, -Z_SCORE_CLAMP, Z_SCORE_CLAMP)

    annualized_exponent = _clamp(
        slope * TRADING_DAYS_PER_YEAR,
        -ANNUALIZED_SLOPE_EXPONENT_CLAMP,
        ANNUALIZED_SLOPE_EXPONENT_CLAMP,
    )
    annualized_slope = math.exp(annualized_exponent) - 1.0

    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((log_prices - np.mean(log_prices)) ** 2))
    r_squared = (1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    r_squared = max(0.0, min(1.0, r_squared))

    return TimeframeMetrics(
        label=label,
        lookback=lookback,
        slope=slope,
        intercept=intercept,
        endpoint_index=endpoint_index,
        current_price=current_price,
        fitted_price=fitted_price,
        residual_std=residual_std,
        z_score=z_score,
        annualized_slope=annualized_slope,
        r_squared=r_squared,
    )


def compute_all_timeframe_metrics(
    prices: Sequence[float],
    *,
    lookbacks: dict[str, int] = LOOKBACKS,
) -> dict[str, TimeframeMetrics]:
    """Compute metrics for every configured timeframe with sufficient history."""
    metrics: dict[str, TimeframeMetrics] = {}
    for label, lookback in lookbacks.items():
        m = compute_timeframe_metrics(label, lookback, prices)
        if m is not None:
            metrics[label] = m
    return metrics


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def deviation_points(z_score: float) -> int:
    if z_score <= -2.50:
        return 6
    if z_score <= -2.00:
        return 5
    if z_score <= -1.50:
        return 4
    if z_score <= -1.00:
        return 3
    if z_score <= -0.50:
        return 2
    if z_score <= 0.00:
        return 1
    return 0


def annualized_slope_points(annualized_slope: float) -> int:
    if annualized_slope >= 0.40:
        return 6
    if annualized_slope >= 0.25:
        return 5
    if annualized_slope >= 0.15:
        return 4
    if annualized_slope >= 0.08:
        return 3
    if annualized_slope >= 0.03:
        return 2
    if annualized_slope >= 0.00:
        return 1
    return 0


def r_squared_points(r_squared: float) -> int:
    if r_squared >= 0.80:
        return 3
    if r_squared >= 0.60:
        return 2
    if r_squared >= 0.35:
        return 1
    return 0


def score_timeframe(metrics: TimeframeMetrics, weight: float) -> float:
    points = (
        deviation_points(metrics.z_score)
        + annualized_slope_points(metrics.annualized_slope)
        + r_squared_points(metrics.r_squared)
    )
    return points * weight


def compute_total_score(
    metrics: dict[str, TimeframeMetrics],
    *,
    weights: dict[str, float] = TIMEFRAME_WEIGHTS,
) -> float:
    return sum(score_timeframe(m, weights.get(label, 1.0)) for label, m in metrics.items())


# ---------------------------------------------------------------------------
# Entry requirements
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EntryRequirements:
    minimum_score: float = 30.0
    minimum_valid_timeframes: int = 6
    minimum_below_line_timeframes: int = 2
    minimum_positive_slope_timeframes: int = 4
    require_positive_long_term_slope: bool = True
    long_term_lookbacks: tuple[str, ...] = LONG_TERM_LOOKBACKS


DEFAULT_ENTRY_REQUIREMENTS = EntryRequirements()


def evaluate_entry_requirements(
    metrics: dict[str, TimeframeMetrics],
    total_score: float,
    current_price: float,
    requirements: EntryRequirements = DEFAULT_ENTRY_REQUIREMENTS,
) -> bool:
    if len(metrics) < requirements.minimum_valid_timeframes:
        return False

    below_line = sum(1 for m in metrics.values() if current_price < m.fitted_price)
    if below_line < requirements.minimum_below_line_timeframes:
        return False

    positive_slope = sum(1 for m in metrics.values() if m.slope > 0)
    if positive_slope < requirements.minimum_positive_slope_timeframes:
        return False

    if requirements.require_positive_long_term_slope:
        has_positive_long_term = any(
            metrics[label].slope > 0
            for label in requirements.long_term_lookbacks
            if label in metrics
        )
        if not has_positive_long_term:
            return False

    if total_score < requirements.minimum_score:
        return False

    return True


# ---------------------------------------------------------------------------
# Locked target selection + projection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LockedTarget:
    label: str
    lookback: int
    slope: float
    intercept: float
    endpoint_index: int
    r_squared: float
    entry_price: float
    entry_target_price: float
    entry_upside: float


@dataclass(frozen=True)
class TargetRequirements:
    minimum_r_squared: float = 0.30
    require_positive_slope: bool = True
    require_price_below_target_line: bool = True
    minimum_entry_upside: float = 0.01
    maximum_entry_upside: float = 0.50
    candidate_labels: tuple[str, ...] = TARGET_LOOKBACKS


DEFAULT_TARGET_REQUIREMENTS = TargetRequirements()


def select_locked_target(
    metrics: dict[str, TimeframeMetrics],
    current_price: float,
    requirements: TargetRequirements = DEFAULT_TARGET_REQUIREMENTS,
) -> LockedTarget | None:
    """Choose the locked exit target per Draco's eligibility + tie-break rules.

    Eligibility: R² >= minimum, positive slope, price below the target line,
    upside within [minimum, maximum]. Tie-break: highest R² -> highest upside
    -> longest lookback.
    """
    candidates: list[tuple[TimeframeMetrics, float]] = []
    for label in requirements.candidate_labels:
        m = metrics.get(label)
        if m is None:
            continue
        if m.r_squared < requirements.minimum_r_squared:
            continue
        if requirements.require_positive_slope and m.slope <= 0:
            continue
        if requirements.require_price_below_target_line and current_price >= m.fitted_price:
            continue
        if current_price <= 0:
            continue
        upside = m.fitted_price / current_price - 1.0
        if upside < requirements.minimum_entry_upside or upside > requirements.maximum_entry_upside:
            continue
        candidates.append((m, upside))

    if not candidates:
        return None

    best_m, best_upside = max(candidates, key=lambda item: (item[0].r_squared, item[1], item[0].lookback))

    return LockedTarget(
        label=best_m.label,
        lookback=best_m.lookback,
        slope=best_m.slope,
        intercept=best_m.intercept,
        endpoint_index=best_m.endpoint_index,
        r_squared=best_m.r_squared,
        entry_price=current_price,
        entry_target_price=best_m.fitted_price,
        entry_upside=best_upside,
    )


def project_locked_target(target: LockedTarget, trading_days_held: int) -> float:
    """Project the locked regression line forward `trading_days_held` sessions.

    The target's slope/intercept/endpoint_index are frozen at entry and never
    recalculated; only the projection index advances.
    """
    projected_index = target.endpoint_index + trading_days_held
    exponent = _clamp(
        target.intercept + target.slope * projected_index,
        -EXP_OVERFLOW_GUARD,
        EXP_OVERFLOW_GUARD,
    )
    return math.exp(exponent)


# ---------------------------------------------------------------------------
# Mean-reversion fitness scoring
#
# Answers a narrower question than the live entry logic above: not "would
# Draco's full candidate-ranking/entry/exit/circuit-breaker strategy have
# traded this ticker profitably," but "when this ticker has dipped notably
# below its own regression trend line, how often and how far does it
# actually come back?" That's exactly what the per-day (z_score,
# fitted_price) already computed above measures -- this section scores a
# ticker's own historical tendency to close that gap, instead of simulating
# a multi-ticker portfolio trading it. No universe, no portfolio, no
# stop-loss/circuit-breaker/risk-off simulation: one ticker's own price
# history in, one score out.
# ---------------------------------------------------------------------------

# "Meaningfully below trend" -- reuses deviation_points' -1.0 breakpoint (3 of
# its 6 points) rather than inventing a new threshold.
DIP_Z_THRESHOLD = -1.0

# How many forward trading days to check whether a dip closed, per timeframe.
# Shorter timeframes get a shorter recovery window; longer ones get more
# room, capped at 63 (~1 quarter) so even a 5Y lookback's dip doesn't require
# waiting years to learn whether it resolved.
REVERSION_HORIZON_DAYS: dict[str, int] = {
    "1M": 5,
    "3M": 10,
    "6M": 15,
    "1Y": 21,
    "2Y": 42,
    "3Y": 63,
    "4Y": 63,
    "5Y": 63,
}

# Bars between the dip-detection day and the simulated "entry" used to judge
# the outcome -- the same one-bar fill-lag convention used elsewhere in this
# codebase's fitness scoring (the decision day's last price point is day
# D-1's close; the earliest fill that doesn't see the future is day D's
# close), duplicated locally so this module stays dependency-free.
REVERSION_FILL_LAG_DAYS = 1

# normalize_reversion_quality()'s sigmoid center/scale. Empirically
# calibrated (2026-10-26) against an 88-ticker, sector/cap-diverse basket's
# actual totalScore distribution -- not a guess. The original 0.40/0.35
# assumed a typical totalScore far higher than what real tickers produce
# (median was ~0.005, since most tickers aren't currently dipped on any
# timeframe and so score exactly/near 0 -- correctly, since there's no
# live opportunity to measure), which clustered ~82% of that basket into
# "poor" and made the label essentially non-discriminating. 0.19/0.20 keeps
# every zero-signal ticker in "poor" (no current dip really is "poor" for a
# dip-timing score) while spreading tickers with a real, currently-active
# dip across weak/moderate/good/strong based on how large and well-evidenced
# that dip's reversion history is. Revisit the same way if the real-world
# totalScore distribution drifts (e.g. after a market regime change, or
# after further changes to draco_math.py's dip-episode/confidence logic).
REVERSION_QUALITY_CENTER = 0.19
REVERSION_QUALITY_SCALE = 0.20


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _walk_forward_split_index(n_rows: int, train_fraction: float) -> int:
    """Train/test boundary for walk-forward validation: prices[:split] is
    train, prices[split:] is test. Clamped so both sides are non-empty
    whenever there are at least 2 rows. Same scheme used elsewhere in this
    codebase's fitness scoring, duplicated locally so this module stays
    dependency-free."""
    if n_rows <= 1:
        return n_rows
    split = int(round(n_rows * train_fraction))
    return min(max(split, 1), n_rows - 1)


def fit_label(score: float) -> str:
    if score >= 0.80:
        return "strong"
    if score >= 0.65:
        return "good"
    if score >= 0.50:
        return "moderate"
    if score >= 0.35:
        return "weak"
    return "poor"


def normalize_reversion_quality(total_score: float) -> float:
    return _clamp(_sigmoid((total_score - REVERSION_QUALITY_CENTER) / REVERSION_QUALITY_SCALE), 0.0, 1.0)


@dataclass(frozen=True)
class DipEpisode:
    """One maximal run of consecutive below-threshold days, judged only at
    its first ("anchor") day -- the days after that overlap the same
    recovery window and aren't independent evidence."""
    anchor_index: int
    z_score: float
    fitted_price_at_anchor: float
    reverted: bool | None  # None when there isn't enough forward data yet to judge


@dataclass(frozen=True)
class ReversionStats:
    label: str
    episode_count: int  # episodes with a judged (non-None) outcome
    reverted_count: int
    reversion_probability: float


def find_dip_episodes(
    z_scores: Sequence[float],
    fitted_prices: Sequence[float],
    full_prices: Sequence[float],
    *,
    anchor_offset: int,
    horizon_days: int,
    fill_lag_days: int = REVERSION_FILL_LAG_DAYS,
    dip_threshold: float = DIP_Z_THRESHOLD,
    slopes: Sequence[float] | None = None,
) -> list[DipEpisode]:
    """Scan `z_scores`/`fitted_prices` (positionally aligned, starting at
    absolute day `anchor_offset` within `full_prices`) for dip episodes and
    judge each one: did price close at or above that day's fitted trend line
    within `horizon_days` trading days, starting `fill_lag_days` days after
    the dip was detected?

    When `slopes` is given (one value per z-score), a day only counts as a
    dip if this timeframe's own trend line is upward-sloping there too --
    mirroring the live strategy's own positive-slope requirement (see
    EntryRequirements.minimum_positive_slope_timeframes). A decline against a
    *downward*-sloping line isn't a setup Draco would ever trade, so it
    shouldn't count as reversion evidence either."""
    episodes: list[DipEpisode] = []
    in_dip = False
    for i, z in enumerate(z_scores):
        is_uptrend = slopes is None or slopes[i] > 0.0
        is_dip_day = (z <= dip_threshold) and is_uptrend
        if is_dip_day and not in_dip:
            exit_idx = anchor_offset + i + fill_lag_days + horizon_days
            reverted: bool | None = None
            if 0 <= exit_idx < len(full_prices):
                reverted = float(full_prices[exit_idx]) >= float(fitted_prices[i])
            episodes.append(DipEpisode(
                anchor_index=anchor_offset + i,
                z_score=float(z),
                fitted_price_at_anchor=float(fitted_prices[i]),
                reverted=reverted,
            ))
        in_dip = is_dip_day
    return episodes


def summarize_reversion(label: str, episodes: Sequence[DipEpisode]) -> ReversionStats:
    judged = [e for e in episodes if e.reverted is not None]
    reverted_count = sum(1 for e in judged if e.reverted)
    probability = (reverted_count / len(judged)) if judged else 0.0
    return ReversionStats(
        label=label,
        episode_count=len(judged),
        reverted_count=reverted_count,
        reversion_probability=probability,
    )


def rolling_timeframe_series(
    prices: np.ndarray,
    label: str,
    lookback: int,
    *,
    start_index: int,
) -> tuple[list[float], list[float], list[float]]:
    """z_score, fitted_price, and annualized_slope for every day from
    `start_index` through the end of `prices`, recomputing the trailing
    `lookback`-bar regression fresh each day (no state carried across days,
    matching the live scan's own per-day recompute)."""
    z_scores: list[float] = []
    fitted_prices: list[float] = []
    slopes: list[float] = []
    for i in range(start_index, len(prices)):
        metrics = compute_timeframe_metrics(label, lookback, prices[: i + 1])
        if metrics is None:
            z_scores.append(0.0)
            fitted_prices.append(float(prices[i]))
            slopes.append(0.0)
        else:
            z_scores.append(metrics.z_score)
            fitted_prices.append(metrics.fitted_price)
            slopes.append(metrics.annualized_slope)
    return z_scores, fitted_prices, slopes


@dataclass(frozen=True)
class TimeframeReversionProfile:
    label: str
    lookback: int
    weight: float
    z_score_now: float
    distance_to_mean_now: float  # max(fitted/current - 1, 0.0)
    reversion_probability_train: float
    reversion_probability_test: float | None
    episode_count_train: int
    episode_count_test: int
    expected_value: float  # scoring probability * distance_to_mean_now, 0 unless currently dipped


def compute_timeframe_reversion_profile(
    prices: np.ndarray,
    label: str,
    lookback: int,
    weight: float,
    *,
    start_index: int,
    split_index: int,
    horizon_days: int,
) -> TimeframeReversionProfile | None:
    """One timeframe's full mean-reversion profile for a single ticker:
    train-period reversion probability, an out-of-sample re-check on the
    test period (mirroring this codebase's other walk-forward validations),
    and today's live dip distance. Returns None when there isn't enough
    history to warm up this lookback."""
    if len(prices) <= lookback:
        return None
    scan_start = max(lookback, start_index)
    if scan_start >= len(prices):
        return None

    full_z, full_fitted, full_slopes = rolling_timeframe_series(prices, label, lookback, start_index=scan_start)
    # full_z[i] / full_fitted[i] / full_slopes[i] correspond to absolute day `scan_start + i`.

    train_end = max(0, min(len(full_z), split_index - scan_start))
    train_episodes = find_dip_episodes(
        full_z[:train_end], full_fitted[:train_end], prices,
        anchor_offset=scan_start, horizon_days=horizon_days,
        slopes=full_slopes[:train_end],
    )
    train_stats = summarize_reversion(label, train_episodes)

    test_stats: ReversionStats | None = None
    if train_end < len(full_z):
        test_episodes = find_dip_episodes(
            full_z[train_end:], full_fitted[train_end:], prices,
            anchor_offset=scan_start + train_end, horizon_days=horizon_days,
            slopes=full_slopes[train_end:],
        )
        test_stats = summarize_reversion(label, test_episodes)

    # Prefer the out-of-sample (test-period) reversion rate when there's any
    # test-period evidence to measure it from -- same "trust the held-out
    # result, not the in-sample one" principle this codebase already applies
    # to Force/Aptet/Draco's grid-search walk-forward validation.
    scoring_probability = (
        test_stats.reversion_probability
        if test_stats is not None and test_stats.episode_count > 0
        else train_stats.reversion_probability
    )

    now_z = full_z[-1]
    now_fitted = full_fitted[-1]
    now_current = float(prices[-1])
    distance_to_mean_now = max(0.0, now_fitted / now_current - 1.0) if now_current > 0 else 0.0
    is_dipped_now = now_z < 0.0
    expected_value = (scoring_probability * distance_to_mean_now) if is_dipped_now else 0.0

    return TimeframeReversionProfile(
        label=label,
        lookback=lookback,
        weight=weight,
        z_score_now=now_z,
        distance_to_mean_now=distance_to_mean_now,
        reversion_probability_train=train_stats.reversion_probability,
        reversion_probability_test=(test_stats.reversion_probability if test_stats is not None else None),
        episode_count_train=train_stats.episode_count,
        episode_count_test=(test_stats.episode_count if test_stats is not None else 0),
        expected_value=expected_value,
    )


def _reversion_confidence(profiles: Sequence[TimeframeReversionProfile], *, timeframe_count: int) -> float:
    total_test_episodes = sum(p.episode_count_test for p in profiles)
    sample_component = _clamp(math.sqrt(total_test_episodes / 20.0), 0.0, 1.0)

    stability_terms = [
        1.0 - min(1.0, abs(p.reversion_probability_train - p.reversion_probability_test))
        for p in profiles
        if p.episode_count_train > 0 and p.episode_count_test > 0 and p.reversion_probability_test is not None
    ]
    stability_component = (sum(stability_terms) / len(stability_terms)) if stability_terms else 0.4

    coverage_component = _clamp(len(profiles) / float(max(1, timeframe_count)), 0.0, 1.0)

    return _clamp(
        (sample_component * 0.45) + (stability_component * 0.35) + (coverage_component * 0.20),
        0.0,
        1.0,
    )


@dataclass(frozen=True)
class MeanReversionFitness:
    ticker: str
    total_score: float
    raw_score: float  # sigmoid-normalized quality in [0, 1], before the confidence penalty
    confidence: float
    score: float  # confidence-penalized final score in [0, 1]
    label: str
    profiles: list[TimeframeReversionProfile]
    start_index: int  # absolute position in `prices` where the scored window begins
    split_index: int  # absolute position where the train/test walk-forward split falls


def compute_mean_reversion_fitness(
    prices: np.ndarray,
    ticker: str,
    *,
    start_index: int = 0,
    lookbacks: dict[str, int] = LOOKBACKS,
    weights: dict[str, float] = TIMEFRAME_WEIGHTS,
    horizons: dict[str, int] = REVERSION_HORIZON_DAYS,
    train_fraction: float = 0.70,
) -> MeanReversionFitness | None:
    """Score how good a Draco mean-reversion candidate `ticker` is *right
    now*: for each configured timeframe, how reliably (out of sample) this
    ticker has historically closed a meaningful gap below its own trend
    line, multiplied by how large that gap is today -- summed across
    timeframes using Draco's existing longer-is-weighted-more scheme.
    Returns None when there isn't enough history for even the shortest
    configured timeframe."""
    longest_lookback = max(lookbacks.values())
    effective_start = max(longest_lookback, start_index)
    if effective_start >= len(prices):
        return None

    scoring_length = len(prices) - effective_start
    split_index = effective_start + _walk_forward_split_index(scoring_length, train_fraction)

    profiles: list[TimeframeReversionProfile] = []
    for label, lookback in lookbacks.items():
        profile = compute_timeframe_reversion_profile(
            prices, label, lookback, weights.get(label, 1.0),
            start_index=effective_start, split_index=split_index,
            horizon_days=horizons.get(label, 21),
        )
        if profile is not None:
            profiles.append(profile)
    if not profiles:
        return None

    total_score = sum(p.expected_value * p.weight for p in profiles)
    raw_score = normalize_reversion_quality(total_score)
    confidence = _reversion_confidence(profiles, timeframe_count=len(lookbacks))
    score = _clamp(raw_score * confidence + raw_score * (1.0 - confidence) * 0.5, 0.0, 1.0)

    return MeanReversionFitness(
        ticker=ticker,
        total_score=total_score,
        raw_score=raw_score,
        confidence=confidence,
        score=score,
        label=fit_label(score),
        profiles=profiles,
        start_index=effective_start,
        split_index=split_index,
    )
