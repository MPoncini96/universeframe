import pandas as pd
import pytest

from universeframe import (
    AdaptiveUniverseScorer,
    CompositeScorer,
    WeightedScorer,
    build_universe,
    get_scorer,
    register_scorer,
)
from universeframe.engine import _SCORER_REGISTRY


def _vectors_df(rows: dict[str, dict]) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "ticker"
    df["feature_coverage_ratio"] = df.get("feature_coverage_ratio", 1.0)
    return df


class _RankScorer:
    """Stand-in for a downloaded algorithm's own scorer plugin."""

    feature_keys = ["rank"]

    def score(self, candidate: dict) -> float:
        return candidate.get("rank", 0.0)


@pytest.fixture
def registered_rank_scorer():
    key = "test_third_party_algo"
    register_scorer(key, _RankScorer())
    try:
        yield key
    finally:
        _SCORER_REGISTRY.pop(key, None)


def test_a_newly_registered_algorithm_is_ranked_by_the_engine(registered_rank_scorer):
    df = _vectors_df({
        "LOW": {"rank": 0.1},
        "HIGH": {"rank": 0.9},
        "MID": {"rank": 0.5},
    })
    result = build_universe(df, algorithm=registered_rank_scorer, size=3)
    assert list(result.index) == ["HIGH", "MID", "LOW"]


def test_get_scorer_returns_the_exact_registered_instance():
    scorer = _RankScorer()
    register_scorer("test_get_scorer_algo", scorer)
    try:
        assert get_scorer("test_get_scorer_algo") is scorer
    finally:
        _SCORER_REGISTRY.pop("test_get_scorer_algo", None)


def test_get_scorer_raises_for_unknown_algorithm():
    with pytest.raises(ValueError, match="definitely_not_registered"):
        get_scorer("definitely_not_registered")


def test_build_universe_requires_exactly_one_of_scorer_or_algorithm():
    df = _vectors_df({"AAA": {"rank": 0.5}})
    with pytest.raises(ValueError):
        build_universe(df)
    with pytest.raises(ValueError):
        build_universe(df, scorer=_RankScorer(), algorithm="x")


def test_feature_coverage_filter_drops_thin_candidates():
    df = _vectors_df({
        "GOOD": {"momentum_score": 0.9, "feature_coverage_ratio": 0.95},
        "THIN": {"momentum_score": 0.9, "feature_coverage_ratio": 0.10},
    })
    result = build_universe(df, scorer=CompositeScorer("momentum_score"), size=5)
    assert list(result.index) == ["GOOD"]


def test_sector_filter_matches_case_insensitively():
    df = _vectors_df({
        "TECH": {"momentum_score": 0.5, "sector_name": "Information Technology"},
        "ENERGY": {"momentum_score": 0.9, "sector_name": "Energy"},
    })
    result = build_universe(df, scorer=CompositeScorer("momentum_score"), sectors=["information technology"], size=5)
    assert list(result.index) == ["TECH"]


def test_tags_blend_with_algorithm_score():
    df = _vectors_df({
        "AAA": {"momentum_score": 1.0, "drawdown_state_score": 0.0},
    })
    result = build_universe(df, scorer=CompositeScorer("momentum_score"), tags=["low drawdown"], size=1)
    # algo_score=1.0 (momentum), tag_score=1.0 (1 - drawdown_state_score=0) -> relevance=1.0
    assert result.loc["AAA", "relevance"] == pytest.approx(1.0)
    assert result.loc["AAA", "tag_score"] == pytest.approx(1.0)


def test_weighted_scorer_renormalizes_over_present_weights():
    scorer = WeightedScorer({"momentum_score": 0.5, "trend_quality_score": 0.5})
    candidate = {"composite_scores": {"momentum_score": 1.0, "trend_quality_score": None}, "momentum_score": 1.0}
    assert scorer.score(candidate) == pytest.approx(1.0)


def test_in_repo_scorers_satisfy_the_protocol():
    assert isinstance(CompositeScorer("momentum_score"), AdaptiveUniverseScorer)
    assert isinstance(WeightedScorer({"momentum_score": 1.0}), AdaptiveUniverseScorer)
