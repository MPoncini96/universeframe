"""Shared walk-forward fitness math used by both Force and Draco.

A verbatim port of Monstra-Worker's ``scripts/stock_fitness_common.py`` - the
same per-combo scoring formula, sample-confidence estimate, and
walk-forward-aware ``calculate_algorithm_fit`` aggregation every algorithm
family's fitness analyzer shares, so Force and Draco (and, if added later,
any other family) can't silently drift apart. The one thing intentionally
left out is ``load_reference_universe``, which reads a persisted default
universe from Monstra's Postgres - standalone runs always pass an explicit
universe instead.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np
import pandas as pd

COMBO_QUALITY_CENTER = 0.35
COMBO_QUALITY_SCALE = 0.22
PROFILE_BREADTH_RELATIVE_THRESHOLD = 0.70
PROFILE_BREADTH_FLOOR = 0.55
RECOMMENDED_REGION_RELATIVE_THRESHOLD = 0.85
RECOMMENDED_REGION_CONFIDENCE_FLOOR = 0.35
UNIVERSE_IDENTITY_VERSION = 1
FIT_SCOPE = "universe_conditioned"
CONFIDENCE_ADJUSTMENT_METHOD = "conservative_penalty"

# How many bars after the ranking decision the simulated trade is assumed to
# actually fill. The ranking at decision index D uses trailing returns
# computed from prices[:D], so the last price point feeding the decision is
# D-1's close. entry_lag_days=1 shifts the fill to D's close (one bar after
# the last data point the ranking saw) - the earliest fill assumption that
# doesn't require seeing the future relative to the decision.
DEFAULT_EXECUTION_LAG_DAYS = 1

# Walk-forward out-of-sample validation of the grid-search-selected combo.
#
# Grid-searching N (lookback, portfolio_size)-style combos and reporting the
# single best in-sample result is a classic multiple-comparisons setup: with
# enough combos, some will look good on noise alone. WALK_FORWARD_TRAIN_FRACTION
# splits the requested date range into an earlier train slice (used to select
# a combo) and a later held-out test slice (used to validate that selection
# honestly). 0.70 is a conventional walk-forward split; it is not empirically
# tuned for this codebase.
WALK_FORWARD_TRAIN_FRACTION = 0.70


def clean_ticker(value: Any, fallback: str | None = None) -> str:
    token = str(value or fallback or "").strip().upper()
    return token or str(fallback or "").strip().upper()


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def forward_return(
    prices: pd.DataFrame,
    decision_idx: int,
    ticker: str,
    horizon_days: int,
    *,
    entry_lag_days: int = DEFAULT_EXECUTION_LAG_DAYS,
) -> float | None:
    """Shared forward-return calc used by every algorithm family."""
    entry_idx = decision_idx - 1 + entry_lag_days
    exit_idx = entry_idx + horizon_days
    if entry_idx < 0 or exit_idx >= len(prices.index):
        return None
    start_price = safe_float(prices.iloc[entry_idx].get(ticker))
    end_price = safe_float(prices.iloc[exit_idx].get(ticker))
    if start_price is None or end_price is None or start_price <= 0.0:
        return None
    return (end_price / start_price) - 1.0


def max_drawdown(equity_curve: list[float]) -> float:
    """Shared max-drawdown calc used by every algorithm family."""
    if not equity_curve:
        return 0.0
    equity = pd.Series(equity_curve, dtype=float)
    running_max = equity.cummax()
    drawdown = (equity / running_max) - 1.0
    return float(drawdown.min())


def compute_walk_forward_split_index(n_rows: int, train_fraction: float = WALK_FORWARD_TRAIN_FRACTION) -> int:
    """Positional row index marking the train/test boundary for walk-forward
    validation: prices.iloc[:split] is train, prices.iloc[split:] is test.
    Clamped so both sides are non-empty whenever there are at least 2 rows."""
    if n_rows <= 1:
        return n_rows
    split = int(round(n_rows * train_fraction))
    return min(max(split, 1), n_rows - 1)


def slice_test_prices_with_lookback_buffer(prices: pd.DataFrame, split_idx: int, lookback_days: int) -> pd.DataFrame:
    """The test slice for out-of-sample validation needs `lookback_days` of
    pre-split history prepended as trailing-window warmup - otherwise the
    first eligible decision day inside the slice would be `lookback_days`
    calendar days into the test period instead of right at the split."""
    start = max(0, split_idx - max(0, lookback_days))
    return prices.iloc[start:]


def date_bounds(prices: pd.DataFrame, start_idx: int, end_idx_inclusive: int) -> dict[str, Any] | None:
    """Safe {"start", "end"} date-range dict for a positional slice of
    `prices`, or None when the slice is empty."""
    if end_idx_inclusive < start_idx or start_idx < 0 or end_idx_inclusive >= len(prices.index):
        return None
    return {
        "start": str(prices.index[start_idx].date()),
        "end": str(prices.index[end_idx_inclusive].date()),
    }


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(value)))


def normalize_combo_quality(raw_fitness_score: float) -> float:
    return clamp01(sigmoid((float(raw_fitness_score) - COMBO_QUALITY_CENTER) / COMBO_QUALITY_SCALE))


def calculate_combo_sample_confidence(
    activation_count: int,
    eligible_days: int,
    horizon_coverage: dict[int, float],
    *,
    activation_episode_count: int | None = None,
) -> float:
    """Shared per-combo sample-confidence estimate used by every algorithm
    family.

    activation_episode_count -- the number of contiguous activation runs
    (how many times the strategy transitioned from inactive to active), as
    opposed to activation_count's raw day count. Forward-return samples from
    consecutive days inside the same activation run overlap heavily, so they
    are not independent observations. When provided, the activation
    component is instead driven by the episode count using the same
    20-for-full-confidence calibration as before."""
    effective_activation_count = (
        activation_episode_count if activation_episode_count is not None else activation_count
    )
    activation_component = clamp01(math.sqrt(max(0.0, effective_activation_count) / 20.0))
    eligible_component = clamp01(math.sqrt(max(0.0, eligible_days) / 126.0))
    horizon_component = float(np.mean(list(horizon_coverage.values()))) if horizon_coverage else 0.0
    return clamp01((activation_component * 0.55) + (eligible_component * 0.25) + (horizon_component * 0.20))


def score_combo(metrics: dict[str, Any], score_mode: str) -> float:
    """Shared per-combo fitness-score formula used by every algorithm
    family."""
    activation_rate = float(metrics["activationRate"])
    hit_rate = float(metrics["hitRate1D"])
    avg_return_1d = float(metrics["avgForwardReturn1D"])
    return_lift_1d = float(metrics["returnLift1D"])
    shadow_total_return = float(metrics["shadowTotalReturn"])
    shadow_sharpe = float(metrics["shadowSharpeLike"])
    shadow_drawdown = float(metrics["shadowMaxDrawdown"])
    confidence = float(metrics["sampleConfidence"])

    precision_score = ((hit_rate - 0.5) * 2.5) + (avg_return_1d * 18.0) + (return_lift_1d * 15.0)
    coverage_score = activation_rate * 1.5
    shadow_score = (shadow_total_return * 3.0) + (shadow_sharpe * 0.15) + (shadow_drawdown * 1.2)

    if score_mode == "precision":
        raw_score = precision_score + (coverage_score * 0.15)
    elif score_mode == "coverage":
        raw_score = (precision_score * 0.65) + coverage_score + (shadow_score * 0.15)
    elif score_mode == "shadow":
        raw_score = shadow_score + (precision_score * 0.25)
    else:
        raw_score = (precision_score * 0.55) + (coverage_score * 0.35) + (shadow_score * 0.35)
    return raw_score * confidence


def normalize_universe_members(universe: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for member in universe:
        ticker = clean_ticker(member)
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        normalized.append(ticker)
    return sorted(normalized)


def build_universe_identity(universe: list[str]) -> dict[str, Any]:
    members = normalize_universe_members(universe)
    payload = json.dumps(members, separators=(",", ":"))
    return {
        "members": members,
        "size": len(members),
        "hash": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "version": UNIVERSE_IDENTITY_VERSION,
    }


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


def combo_key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["lookbackDays"]), int(row["topN"])


def neighbor_keys(best_row: dict[str, Any], combo_index: dict[tuple[int, int], dict[str, Any]]) -> list[tuple[int, int]]:
    best_lookback = int(best_row["lookbackDays"])
    best_top_n = int(best_row["topN"])
    lookbacks = sorted({int(row["lookbackDays"]) for row in combo_index.values()})
    top_ns = sorted({int(row["topN"]) for row in combo_index.values()})
    lookback_pos = lookbacks.index(best_lookback)
    top_n_pos = top_ns.index(best_top_n)
    neighbors: list[tuple[int, int]] = []
    for delta in (-1, 1):
        if 0 <= lookback_pos + delta < len(lookbacks):
            neighbors.append((lookbacks[lookback_pos + delta], best_top_n))
        if 0 <= top_n_pos + delta < len(top_ns):
            neighbors.append((best_lookback, top_ns[top_n_pos + delta]))
    deduped: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for key in neighbors:
        if key in combo_index and key not in seen:
            seen.add(key)
            deduped.append(key)
    return deduped


def calculate_algorithm_fit(
    combo_rows: list[dict[str, Any]],
    *,
    out_of_sample_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """out_of_sample_profile -- an _analyze_combo-shaped metrics dict for the
    grid-search winner, re-evaluated on data the grid search never saw. When
    provided, its comboQuality/fitnessScore replace the in-sample best
    combo's in the weighted raw_score formula below - this is the guard
    against picking the literal best of N grid-searched combos and reporting
    its own in-sample number as if it were unbiased, the classic
    multiple-comparisons failure mode of any "best of a grid search" score.
    Every other raw_score term (median/upper-quartile/parameter-stability)
    is left on the full in-sample grid, since those describe the shape of
    the parameter surface rather than a single cherry-picked point."""
    if not combo_rows:
        return {
            "score": 0.0,
            "rawScore": 0.0,
            "label": "poor",
            "confidence": 0.0,
            "confidenceAdjustment": CONFIDENCE_ADJUSTMENT_METHOD,
            "parameterStability": 0.0,
            "profileBreadth": 0.0,
            "bestProfile": None,
            "recommendedRegion": {"lookbackDays": [], "topNs": []},
            "outOfSampleValidated": False,
            "calibration": {
                "comboQualityCenter": COMBO_QUALITY_CENTER,
                "comboQualityScale": COMBO_QUALITY_SCALE,
                "bestRawFitnessScore": 0.0,
                "bestComboQuality": 0.0,
                "medianComboQuality": 0.0,
                "upperQuartileComboQuality": 0.0,
                "outOfSampleComboQuality": 0.0,
                "outOfSampleRawFitnessScore": 0.0,
            },
        }

    enriched_rows: list[dict[str, Any]] = []
    for row in combo_rows:
        copied = dict(row)
        copied.setdefault("sampleConfidence", 0.0)
        copied["comboQuality"] = float(copied.get("comboQuality", normalize_combo_quality(float(copied.get("fitnessScore", 0.0)))))
        enriched_rows.append(copied)

    ranked = sorted(
        enriched_rows,
        key=lambda item: (float(item["comboQuality"]), float(item.get("fitnessScore", 0.0))),
        reverse=True,
    )
    best = ranked[0]
    best_quality = float(best["comboQuality"])
    best_raw_fitness_score = float(best.get("fitnessScore", 0.0))

    if out_of_sample_profile is not None:
        oos_raw_fitness_score = float(out_of_sample_profile.get("fitnessScore", 0.0))
        oos_quality = float(
            out_of_sample_profile.get("comboQuality", normalize_combo_quality(oos_raw_fitness_score))
        )
    else:
        oos_raw_fitness_score = best_raw_fitness_score
        oos_quality = best_quality

    qualities = [float(row["comboQuality"]) for row in enriched_rows]
    median_quality = float(np.median(qualities)) if qualities else 0.0
    upper_quartile_quality = float(np.quantile(qualities, 0.75)) if qualities else 0.0

    combo_index = {combo_key(row): row for row in enriched_rows}
    direct_neighbors = neighbor_keys(best, combo_index)
    if direct_neighbors:
        neighbor_ratios: list[float] = []
        for key in direct_neighbors:
            neighbor = combo_index[key]
            neighbor_quality = float(neighbor["comboQuality"])
            retained_quality = clamp01(neighbor_quality / max(best_quality, 1e-9)) ** 2
            neighbor_ratios.append(retained_quality)
        parameter_stability = float(np.mean(neighbor_ratios)) if neighbor_ratios else 0.0
    else:
        parameter_stability = best_quality

    breadth_threshold = max(PROFILE_BREADTH_FLOOR, best_quality * PROFILE_BREADTH_RELATIVE_THRESHOLD)
    breadth_weights = [max(0.05, float(row.get("sampleConfidence", 0.0))) for row in enriched_rows]
    strong_weights = [weight for row, weight in zip(enriched_rows, breadth_weights) if float(row["comboQuality"]) >= breadth_threshold]
    profile_breadth = (sum(strong_weights) / sum(breadth_weights)) if breadth_weights and sum(breadth_weights) > 0 else 0.0

    sufficiently_sampled_profiles = [row for row in enriched_rows if float(row.get("sampleConfidence", 0.0)) >= 0.55]
    average_profile_confidence = float(np.mean([float(row.get("sampleConfidence", 0.0)) for row in enriched_rows])) if enriched_rows else 0.0
    max_eligible_days = max(int(row.get("eligibleDays", 0)) for row in enriched_rows)
    eligible_component = clamp01(math.sqrt(max_eligible_days / 252.0))
    activation_depth = clamp01(
        math.sqrt(
            sum(min(int(row.get("activationEpisodeCount", row.get("activationCount", 0))), 20) for row in enriched_rows)
            / float(max(1, len(enriched_rows)) * 20)
        )
    )
    sampled_profile_component = len(sufficiently_sampled_profiles) / float(len(enriched_rows))
    horizon_component_values: list[float] = []
    for row in enriched_rows:
        counts = row.get("horizonObservationCounts", {})
        activation_count = max(1, int(row.get("activationCount", 0)))
        if counts:
            ratios = [clamp01(int(count) / float(activation_count)) for count in counts.values()]
            horizon_component_values.append(float(np.mean(ratios)))
    horizon_component = float(np.mean(horizon_component_values)) if horizon_component_values else 0.0
    breadth_stability_component = (profile_breadth + parameter_stability) / 2.0
    confidence = clamp01(
        (eligible_component * 0.25)
        + (activation_depth * 0.25)
        + (sampled_profile_component * 0.20)
        + (horizon_component * 0.15)
        + (breadth_stability_component * 0.15)
    )
    confidence = clamp01(confidence * (0.50 + (average_profile_confidence * 0.50)))

    raw_score = clamp01(
        (median_quality * 0.40)
        + (upper_quartile_quality * 0.25)
        + (parameter_stability * 0.20)
        + (oos_quality * 0.15)
    )
    effective_score = clamp01((raw_score * confidence) + (raw_score * (1.0 - confidence) * 0.50))

    region_threshold = max(best_quality * RECOMMENDED_REGION_RELATIVE_THRESHOLD, PROFILE_BREADTH_FLOOR)
    region_rows = [
        row for row in enriched_rows
        if float(row["comboQuality"]) >= region_threshold and float(row.get("sampleConfidence", 0.0)) >= RECOMMENDED_REGION_CONFIDENCE_FLOOR
    ]
    if best not in region_rows:
        region_rows.append(best)

    return {
        "score": float(effective_score),
        "rawScore": float(raw_score),
        "label": fit_label(effective_score),
        "confidence": float(confidence),
        "confidenceAdjustment": CONFIDENCE_ADJUSTMENT_METHOD,
        "parameterStability": float(parameter_stability),
        "profileBreadth": float(profile_breadth),
        "bestProfile": {"lookbackDays": int(best["lookbackDays"]), "topN": int(best["topN"])},
        "recommendedRegion": {
            "lookbackDays": sorted({int(row["lookbackDays"]) for row in region_rows}),
            "topNs": sorted({int(row["topN"]) for row in region_rows}),
        },
        "outOfSampleValidated": out_of_sample_profile is not None,
        "calibration": {
            "comboQualityCenter": COMBO_QUALITY_CENTER,
            "comboQualityScale": COMBO_QUALITY_SCALE,
            "bestRawFitnessScore": float(best_raw_fitness_score),
            "bestComboQuality": float(best_quality),
            "medianComboQuality": float(median_quality),
            "upperQuartileComboQuality": float(upper_quartile_quality),
            "outOfSampleRawFitnessScore": float(oos_raw_fitness_score),
            "outOfSampleComboQuality": float(oos_quality),
        },
    }
