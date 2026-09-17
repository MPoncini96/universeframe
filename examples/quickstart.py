"""Build vectors with vectorframe, then rank an adaptive universe on top.

Run with:  python examples/quickstart.py
"""

import logging

from vectorframe import build_vectors

from universeframe import CompositeScorer, WeightedScorer, build_universe

logging.basicConfig(level=logging.INFO, format="%(message)s")

TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "AVGO", "CRM"]

if __name__ == "__main__":
    vectors = build_vectors(TICKERS, benchmark="SPY", years=6)

    print("\n-- Top 5 by momentum_score alone --")
    momentum_universe = build_universe(vectors, scorer=CompositeScorer("momentum_score"), size=5)
    print(momentum_universe)

    print("\n-- Top 5 by a custom momentum+trend blend, tagged 'quality' --")
    blended_scorer = WeightedScorer({"momentum_score": 0.6, "trend_quality_score": 0.4})
    blended_universe = build_universe(vectors, scorer=blended_scorer, tags=["quality"], size=5)
    print(blended_universe)
