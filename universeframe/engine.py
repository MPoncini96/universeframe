"""
engine.py — the shared, algorithm-agnostic half of adaptive-universe
selection: takes a vectors DataFrame (e.g. from `vectorframe.build_vectors`),
applies feature-coverage/sector/industry filtering, and ranks candidates.

Every algorithm-specific decision — which columns it needs, and how it turns
a candidate into a relevance score — is delegated to an
`AdaptiveUniverseScorer` (see scorer.py for generic examples, or write your
own). This module never branches on an algorithm's identity; a scorer is
just an object with a `feature_keys` list and a `score(candidate) -> float`
method, passed in directly or registered under a name via `register_scorer`.

This is a local, standalone recreation of the shared engine behind Monstra's
adaptive-universe algorithms (Force, Aptet, Draco, Echo) — same filtering and
ranking mechanics, minus the database and minus any specific algorithm's own
tuned scoring formula (those stay proprietary to each bot).
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

import pandas as pd

try:
    from vectorframe.contract import COMPOSITE_SCORES as COMPOSITE_SCORE_COLUMNS
except ImportError:  # pragma: no cover - vectorframe is a required dependency at runtime
    COMPOSITE_SCORE_COLUMNS = (
        "cap_score", "liquidity_score", "momentum_score", "stability_score",
        "aggression_score", "quality_momentum_score", "relative_strength_score",
        "trend_quality_score", "downside_risk_score", "drawdown_state_score",
    )

MIN_FEATURE_COVERAGE = 0.9
DEFAULT_UNIVERSE_SIZE = 10

# Tags that map directly to a composite score column — the same vocabulary
# Monstra's own universe-builder UI exposes.
TAG_TO_COMPOSITE: dict[str, str] = {
    "momentum": "momentum_score",
    "trend": "trend_quality_score",
    "stability": "stability_score",
    "aggression": "aggression_score",
    "quality": "quality_momentum_score",
    "relative strength": "relative_strength_score",
    "downside aware": "downside_risk_score",
    "large cap": "cap_score",
    "high liquidity": "liquidity_score",
}
TAG_LOW_DRAWDOWN = "low drawdown"  # special: 1 - drawdown_state_score
SUPPORTED_TAGS = set(TAG_TO_COMPOSITE) | {TAG_LOW_DRAWDOWN}


# ---------------------------------------------------------------------------
# Plugin interface
# ---------------------------------------------------------------------------

@runtime_checkable
class AdaptiveUniverseScorer(Protocol):
    """Implement this (or just duck-type it) to plug your own relevance
    scoring into the engine. `score()` isn't required to pre-clamp to
    [0, 1] — build_universe() clamps the result regardless."""

    feature_keys: list[str]

    def score(self, candidate: dict[str, Any]) -> float: ...


_SCORER_REGISTRY: dict[str, AdaptiveUniverseScorer] = {}


def register_scorer(algorithm: str, scorer: AdaptiveUniverseScorer) -> None:
    """Register `scorer` under `algorithm`, so build_universe(algorithm=...)
    can look it up by name instead of taking a scorer instance directly."""
    _SCORER_REGISTRY[algorithm] = scorer


def get_scorer(algorithm: str) -> AdaptiveUniverseScorer:
    scorer = _SCORER_REGISTRY.get(algorithm)
    if scorer is None:
        raise ValueError(f"Unknown algorithm: {algorithm!r}")
    return scorer


# ---------------------------------------------------------------------------
# Generic helpers (no algorithm-specific knowledge)
# ---------------------------------------------------------------------------

def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        return f if (f == f) and f != float("inf") and f != float("-inf") else None
    except (TypeError, ValueError):
        return None


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _pick_score(composite: dict[str, Optional[float]], key: str) -> float:
    f = _to_float(composite.get(key))
    return f if f is not None else 0.5


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _tag_score(candidate: dict, tags: list[str]) -> Optional[float]:
    if not tags:
        return None
    composite = candidate["composite_scores"]
    scores: list[float] = []
    for tag in tags:
        tag_norm = tag.strip().lower()
        if tag_norm == TAG_LOW_DRAWDOWN:
            scores.append(1.0 - _pick_score(composite, "drawdown_state_score"))
        elif tag_norm in TAG_TO_COMPOSITE:
            scores.append(_pick_score(composite, TAG_TO_COMPOSITE[tag_norm]))
    return _average(scores) if scores else None


def _row_to_candidate(ticker: str, row: pd.Series) -> dict[str, Any]:
    composite = {c: _to_float(row.get(c)) for c in COMPOSITE_SCORE_COLUMNS}
    candidate: dict[str, Any] = {
        "ticker": ticker,
        "sector": row.get("sector_name") or None,
        "industry": row.get("industry_name") or None,
        "feature_coverage": _to_float(row.get("feature_coverage_ratio")),
        "composite_scores": composite,
    }
    # Expose every other column too, so a scorer can read any raw/normalized
    # feature by name (candidate.get("return_3m_score"), etc.).
    for col, val in row.items():
        if col not in candidate:
            candidate[col] = val
    return candidate


# ---------------------------------------------------------------------------
# Scoring + ranking
# ---------------------------------------------------------------------------

def build_universe(
    vectors_df: pd.DataFrame,
    scorer: Optional[AdaptiveUniverseScorer] = None,
    algorithm: Optional[str] = None,
    tags: Optional[list[str]] = None,
    sectors: Optional[list[str]] = None,
    industries: Optional[list[str]] = None,
    size: int = DEFAULT_UNIVERSE_SIZE,
    min_feature_coverage: float = MIN_FEATURE_COVERAGE,
) -> pd.DataFrame:
    """Score and rank candidates from a vectors DataFrame; return the top
    `size` rows, indexed by ticker, with `relevance` / `algorithm_score` /
    `tag_score` columns.

    Args:
        vectors_df: A DataFrame indexed by ticker, e.g. from
            `vectorframe.build_vectors()`. Must have `feature_coverage_ratio`
            and the composite score columns the scorer/tags need.
        scorer: An AdaptiveUniverseScorer instance to score candidates with.
            Pass either this or `algorithm` (a name registered via
            `register_scorer`), not both.
        algorithm: A registered scorer name (see `register_scorer`).
        tags: Descriptive tags (see TAG_TO_COMPOSITE / SUPPORTED_TAGS) — the
            final relevance is the average of the algorithm score and the
            tag score when tags are given, else just the algorithm score.
        sectors / industries: If given, only candidates matching at least
            one (case-insensitive) survive filtering.
        size: How many tickers to return.
        min_feature_coverage: Candidates below this `feature_coverage_ratio`
            are dropped (default matches Monstra's own threshold: 0.9).
    """
    if (scorer is None) == (algorithm is None):
        raise ValueError("Pass exactly one of scorer= or algorithm=.")
    if scorer is None:
        scorer = get_scorer(algorithm)  # type: ignore[arg-type]

    size = max(1, size)
    tag_list = [t.strip().lower() for t in (tags or []) if t.strip()]
    sector_set = {s.strip().lower() for s in (sectors or []) if s.strip()}
    industry_set = {i.strip().lower() for i in (industries or []) if i.strip()}

    rows = []
    for ticker, row in vectors_df.iterrows():
        candidate = _row_to_candidate(str(ticker), row)
        if (candidate["feature_coverage"] or 0.0) < min_feature_coverage:
            continue
        if sector_set or industry_set:
            sec_match = bool(candidate["sector"]) and candidate["sector"].lower() in sector_set
            ind_match = bool(candidate["industry"]) and candidate["industry"].lower() in industry_set
            if not (sec_match or ind_match):
                continue

        algo_score = _clamp01(scorer.score(candidate))
        tag_score = _tag_score(candidate, tag_list)
        relevance = _clamp01(_average([algo_score, tag_score]) if tag_score is not None else algo_score)
        rows.append({
            "ticker": candidate["ticker"],
            "relevance": relevance,
            "algorithm_score": algo_score,
            "tag_score": tag_score,
        })

    result = pd.DataFrame(rows, columns=["ticker", "relevance", "algorithm_score", "tag_score"])
    result = result.sort_values(
        by=["relevance", "algorithm_score", "ticker"], ascending=[False, False, True]
    )
    return result.set_index("ticker").head(size)
