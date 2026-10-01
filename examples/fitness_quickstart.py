"""Walk-forward Force and Draco fitness scoring, straight from yfinance.

Run with:  python examples/fitness_quickstart.py
"""

import json
import logging

from universeframe.fitness import (
    DEFAULT_DRACO_UNIVERSE,
    DEFAULT_FORCE_UNIVERSE,
    DracoFitnessRequest,
    ForceFitnessRequest,
    analyze_draco_fitness,
    analyze_force_fitness,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")

TICKER = "NVDA"

if __name__ == "__main__":
    print(f"\n-- Force fitness for {TICKER} --")
    force_request = ForceFitnessRequest(
        ticker=TICKER,
        universe=[*DEFAULT_FORCE_UNIVERSE, TICKER],
        start_date="2025-01-01",
    )
    force_result = analyze_force_fitness(force_request)
    print(f"score={force_result['algorithmFit']['score']:.3f}  "
          f"label={force_result['algorithmFit']['label']}  "
          f"bestLookback={force_result['bestCombo']['lookbackDays']}d  "
          f"bestPortfolioSize={force_result['bestCombo']['portfolioSize']}")
    print("per-lookback quality (force_10d/15d/45d/3m equivalents):")
    print(json.dumps(force_result["lookbackQualityByLabel"], indent=2))

    print(f"\n-- Draco fitness for {TICKER} --")
    draco_request = DracoFitnessRequest(
        ticker=TICKER,
        universe=[*DEFAULT_DRACO_UNIVERSE, TICKER],
        start_date="2025-01-01",
        use_market_regime_filter=False,  # faster smoke test; True matches production default
    )
    draco_result = analyze_draco_fitness(draco_request)
    print(f"score={draco_result['algorithmFit']['score']:.3f}  "
          f"label={draco_result['algorithmFit']['label']}  "
          f"bestMinEntryScore={draco_result['bestCombo']['minimumEntryScore']}  "
          f"bestMaxPositions={draco_result['bestCombo']['maxPositions']}")
