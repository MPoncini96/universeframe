"""
scorer.py — generic example `AdaptiveUniverseScorer` implementations.

These are intentionally generic (a single composite column, or a
user-supplied linear blend) — not any specific trading strategy's tuned
formula. Write your own class with a `feature_keys` list and a
`score(candidate) -> float` method, then either pass it directly to
`build_universe(scorer=...)` or register it under a name with
`register_scorer()` for the registry-style API.
"""

from __future__ import annotations

from typing import Any

from .engine import _clamp01, _pick_score


class CompositeScorer:
    """Scores each candidate by a single composite score column, e.g.
    "momentum_score" or "quality_momentum_score" (see vectorframe's
    COMPOSITE_SCORES for the full list)."""

    def __init__(self, composite_key: str):
        self.composite_key = composite_key
        self.feature_keys = [composite_key]

    def score(self, candidate: dict[str, Any]) -> float:
        return _pick_score(candidate["composite_scores"], self.composite_key)


class WeightedScorer:
    """Scores each candidate as a weighted average of arbitrary normalized
    [0, 1] feature or composite columns present on the vectors DataFrame.
    Missing values are skipped and the result is renormalized over the
    weights that were present, mirroring vectorframe's `compute_composite`.

    Example:
        WeightedScorer({"momentum_score": 0.6, "trend_quality_score": 0.4})
    """

    def __init__(self, weights: dict[str, float]):
        self.weights = weights
        self.feature_keys = list(weights)

    def score(self, candidate: dict[str, Any]) -> float:
        total_weight = 0.0
        total_score = 0.0
        for key, weight in self.weights.items():
            val = candidate.get(key)
            if val is None:
                val = candidate["composite_scores"].get(key)
            if val is None:
                continue
            total_weight += weight
            total_score += weight * float(val)
        if total_weight <= 0:
            return 0.5
        return _clamp01(total_score / total_weight)
