"""universeframe.fitness — walk-forward fitness scoring for Force and Draco.

A standalone, open-source recreation of the engine behind Monstra's
``trading.Vector.force_fit_score`` / ``draco_fit_score`` columns.

Force: grid-searches its own tunable parameters (lookback days x portfolio
size), backtests every combo through its real holdings-selection logic
(``bots/alpha1.py``), walk-forward validates the grid-search winner on
held-out data, and reduces the result to a single confidence-penalized fit
score in [0, 1] via the shared ``score_combo`` / ``calculate_algorithm_fit``
math in ``common.py``.

Draco asks a narrower, ticker-only question instead: when this ticker has
dipped notably below its own regression trend line (``draco_math.py``), how
often and how far does it actually come back? It measures that empirically
per timeframe, out of sample, with no universe/portfolio simulation - see
``draco.py`` and ``draco_math.py``'s "Mean-reversion fitness scoring".

What's substituted, for both, is the data source (free yfinance history
here, instead of Monstra's live Alpaca/Postgres pipeline) and anything that
depended on a live database (adaptive-universe overrides, persisted bot
state, DB-sourced config) - a standalone run always starts from empty state.
"""

from .common import (
    CONFIDENCE_ADJUSTMENT_METHOD,
    FIT_SCOPE,
    WALK_FORWARD_TRAIN_FRACTION,
    calculate_algorithm_fit,
    fit_label,
)
from .draco import DracoFitnessRequest, analyze_draco_fitness
from .force import DEFAULT_FORCE_UNIVERSE, ForceFitnessRequest, analyze_force_fitness

__all__ = [
    "analyze_force_fitness",
    "ForceFitnessRequest",
    "DEFAULT_FORCE_UNIVERSE",
    "analyze_draco_fitness",
    "DracoFitnessRequest",
    "calculate_algorithm_fit",
    "fit_label",
    "FIT_SCOPE",
    "WALK_FORWARD_TRAIN_FRACTION",
    "CONFIDENCE_ADJUSTMENT_METHOD",
]
