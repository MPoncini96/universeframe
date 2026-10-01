# universeframe

A local, pandas-native "adaptive universe" ranking engine — filter a vectors
DataFrame by sector/industry/feature coverage, score candidates with your own
pluggable scorer, and get back the top-N tickers as a ranked
`pandas.DataFrame`. Also includes `universeframe.fitness`: a faithful,
open-source port of Force's and Draco's own walk-forward fitness-scoring
engine — see [Fitness scoring](#fitness-scoring-universeframefitness) below.

This is a standalone, open-source recreation of the shared engine behind
[Monstra](https://monstra.bot)'s adaptive-universe algorithms (Force, Aptet,
Draco, Echo). It's a companion to
[vectorframe](https://github.com/MPoncini96/vectorframe): vectorframe builds
the per-stock feature vectors, universeframe ranks and filters them into a
universe.

**What's ported vs. what's not:** the filtering/ranking *mechanics* (feature
coverage threshold, sector/industry matching, tag-score blending, sort order)
are a faithful recreation of Monstra's `adaptive_universe_engine.py` — the
same algorithm-agnostic engine Monstra's own docstrings describe as "the seam
for the download-this-algorithm effort." This repo ships that engine plus a
couple of generic example scorers (`CompositeScorer`, `WeightedScorer`) so
you can plug in your own, **and** — in `universeframe.fitness` — a faithful,
open-source port of Force's and Draco's actual tuned scoring/selection logic
(see below). Aptet's and Echo's own scoring plugins stay proprietary for now,
same as in the production codebase, where the engine and each algorithm's
scoring plugin are already separate modules.

## Fitness scoring (`universeframe.fitness`)

`universeframe.fitness` is a standalone, open-source recreation of the
walk-forward backtest behind Monstra's `trading.Vector.force_fit_score` and
`draco_fit_score` columns. For one ticker, it:

1. Grid-searches each algorithm's own tunable parameters — Force: lookback
   days × portfolio size; Draco: minimum entry score × max positions.
2. Backtests every combo through that algorithm's **real holdings-selection
   logic** — Force's rank-weighted top-N selection with its incumbency-margin
   turnover buffer and kill-switch (ported from `bots/alpha1.py`), Draco's
   multi-timeframe log-linear regression entry/target/exit logic (ported from
   `draco_math.py` / `bots/draco.py`) — not an approximation of it.
3. Walk-forward validates the grid-search winner on data it never saw, so the
   result isn't just "the best of a grid search reporting its own in-sample
   number."
4. Reduces all of that to a single confidence-penalized fit score in `[0, 1]`,
   via the same shared `score_combo` / `calculate_algorithm_fit` formulas
   Monstra's production fitness pipeline uses for every algorithm family
   (`universeframe/fitness/common.py`).

```python
from universeframe.fitness import ForceFitnessRequest, analyze_force_fitness

result = analyze_force_fitness(ForceFitnessRequest(
    ticker="NVDA",
    universe=["XLK", "XLF", "XLV", "XLY", "XLI", "SMH", "NVDA"],
    start_date="2025-01-01",
))
print(result["algorithmFit"]["score"], result["algorithmFit"]["label"])
```

Or from the command line:

```bash
python -m universeframe.fitness.force --ticker NVDA
python -m universeframe.fitness.draco --ticker NVDA
```

What's substituted, relative to production: the data source (free yfinance
history here, via `universeframe/fitness/prices.py`, instead of Monstra's
live Alpaca/Postgres pipeline) and anything that depended on a live
database — adaptive-universe overrides, persisted bot state
(`trading.draco_state`), DB-sourced config, Draco's portfolio-level circuit
breaker (reads live account equity). A standalone run always starts from a
fixed universe and empty state. See `examples/fitness_quickstart.py` for a
fuller example.

**Not investment advice.** A research/educational tool, not a recommendation
to buy or sell any security.

## Install

```bash
pip install -r requirements.txt
# or: pip install -e .
```

(Pulls in [vectorframe](https://github.com/MPoncini96/vectorframe) as a git
dependency — it supplies the vectors DataFrame this package ranks.)

## Quickstart

```python
from vectorframe import build_vectors
from universeframe import build_universe, CompositeScorer

vectors = build_vectors(["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"], years=6)

# Rank by a single composite score
top5 = build_universe(vectors, scorer=CompositeScorer("momentum_score"), size=5)

# Or blend several normalized features/composites yourself
from universeframe import WeightedScorer
my_scorer = WeightedScorer({"momentum_score": 0.6, "trend_quality_score": 0.4})
top5 = build_universe(vectors, scorer=my_scorer, tags=["quality"], sectors=["Information Technology"], size=5)
```

`build_universe` returns a DataFrame indexed by ticker with `relevance`,
`algorithm_score`, and `tag_score` columns, sorted best-first.

See `examples/quickstart.py` for a fuller example.

## Writing your own scorer

A scorer is any object with a `feature_keys` list and a
`score(candidate: dict) -> float` method. `candidate` is a dict built from
one row of the vectors DataFrame: `candidate["composite_scores"]` is a dict
of the 10 composite scores (`momentum_score`, `stability_score`, ...), and
every other column (raw features, normalized `_score` columns, `sector`,
`industry`, `feature_coverage`) is available directly by key too.

```python
class MyScorer:
    feature_keys = ["momentum_score", "volatility_3m_score"]

    def score(self, candidate: dict) -> float:
        momentum = candidate["composite_scores"]["momentum_score"] or 0.5
        vol = candidate.get("volatility_3m_score") or 0.5
        return max(0.0, min(1.0, momentum - 0.3 * vol))

top = build_universe(vectors, scorer=MyScorer(), size=10)
```

You can also register a scorer under a name (mirrors Monstra's own
registry-style call sites) and look it up by string:

```python
from universeframe import register_scorer

register_scorer("my_algo", MyScorer())
top = build_universe(vectors, algorithm="my_algo", size=10)
```

## Tags

`TAG_TO_COMPOSITE` maps human-readable tags (`"momentum"`, `"trend"`,
`"stability"`, `"quality"`, `"relative strength"`, `"downside aware"`,
`"large cap"`, `"high liquidity"`) directly onto composite score columns, plus
a special `"low drawdown"` tag (`1 - drawdown_state_score`). When you pass
`tags=`, the final `relevance` is the average of the algorithm score and the
tag score; without tags, `relevance` is just the algorithm score.

## Development

```bash
pip install -e ".[dev]"
pytest
```

`tests/test_engine.py` proves the same extensibility promise as Monstra's own
engine tests: a brand-new scorer can be registered and ranked without editing
`engine.py` at all.

## License

MIT — see `LICENSE`.
