"""universeframe.fitness — walk-forward fitness scoring for Force and Draco.

A standalone, open-source recreation of the engine behind Monstra's
``trading.Vector.force_fit_score`` / ``draco_fit_score`` columns: for one
ticker, grid-search each algorithm's own tunable parameters (Force: lookback
days x portfolio size; Draco: minimum entry score x max positions),
backtest every combo through that algorithm's real holdings-selection logic,
walk-forward validate the grid-search winner on held-out data, and reduce
the result to a single confidence-penalized fit score in [0, 1].

This is the same shared ``score_combo`` / ``calculate_algorithm_fit`` math
Monstra's production fitness pipeline uses (see ``common.py``), and the same
Force (``bots/alpha1.py``) and Draco (``draco_math.py`` / ``bots/draco.py``)
holdings-selection logic those bots trade live with - not an approximation.
What's substituted is the data source (free yfinance history here, instead
of Monstra's live Alpaca/Postgres pipeline) and anything that depended on a
live database (adaptive-universe overrides, persisted bot state, DB-sourced
config) - a standalone run always starts from a fixed universe and empty
state, same as vectorframe/universeframe's existing scope.
"""

from .common import (
    CONFIDENCE_ADJUSTMENT_METHOD,
    FIT_SCOPE,
    WALK_FORWARD_TRAIN_FRACTION,
    calculate_algorithm_fit,
    fit_label,
)
from .draco import DEFAULT_DRACO_UNIVERSE, DracoFitnessRequest, analyze_draco_fitness
from .force import DEFAULT_FORCE_UNIVERSE, ForceFitnessRequest, analyze_force_fitness

__all__ = [
    "analyze_force_fitness",
    "ForceFitnessRequest",
    "DEFAULT_FORCE_UNIVERSE",
    "analyze_draco_fitness",
    "DracoFitnessRequest",
    "DEFAULT_DRACO_UNIVERSE",
    "calculate_algorithm_fit",
    "fit_label",
    "FIT_SCOPE",
    "WALK_FORWARD_TRAIN_FRACTION",
    "CONFIDENCE_ADJUSTMENT_METHOD",
]
