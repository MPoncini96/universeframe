# universeframe

A local, pandas-native "adaptive universe" ranking engine — filter a vectors
DataFrame by sector/industry/feature coverage, score candidates with your own
pluggable scorer, and get back the top-N tickers as a ranked
`pandas.DataFrame`.

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
for the download-this-algorithm effort." What's **not** included is any
specific bot's tuned scoring formula (Force/Aptet/Draco/Echo's own plugins)
— those stay proprietary to each algorithm, same as in the production
codebase, where the engine and the per-algorithm scoring plugins are already
two separate modules. This repo ships the engine plus a couple of generic
example scorers (`CompositeScorer`, `WeightedScorer`) so you can plug in your
own.

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
