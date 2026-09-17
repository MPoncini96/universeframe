"""universeframe — local, pandas-native adaptive-universe ranking on top of vectorframe."""

from .engine import (
    DEFAULT_UNIVERSE_SIZE,
    MIN_FEATURE_COVERAGE,
    SUPPORTED_TAGS,
    TAG_LOW_DRAWDOWN,
    TAG_TO_COMPOSITE,
    AdaptiveUniverseScorer,
    build_universe,
    get_scorer,
    register_scorer,
)
from .scorer import CompositeScorer, WeightedScorer

__all__ = [
    "build_universe",
    "register_scorer",
    "get_scorer",
    "AdaptiveUniverseScorer",
    "CompositeScorer",
    "WeightedScorer",
    "TAG_TO_COMPOSITE",
    "TAG_LOW_DRAWDOWN",
    "SUPPORTED_TAGS",
    "MIN_FEATURE_COVERAGE",
    "DEFAULT_UNIVERSE_SIZE",
]

__version__ = "0.1.0"
