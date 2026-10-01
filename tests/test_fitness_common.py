import pandas as pd
import pytest

from universeframe.fitness.common import (
    FIT_SCOPE,
    build_universe_identity,
    calculate_algorithm_fit,
    compute_walk_forward_split_index,
    fit_label,
    forward_return,
    max_drawdown,
    normalize_universe_members,
    score_combo,
)


def test_universe_identity_is_order_and_case_independent():
    left = build_universe_identity(["NVDA", "AMD", "MSFT"])
    right = build_universe_identity(["MSFT", "nvda", "AMD", "AMD"])
    assert left == right
    assert normalize_universe_members(["MSFT", "nvda", "AMD", "AMD"]) == ["AMD", "MSFT", "NVDA"]


def test_fit_label_thresholds():
    assert fit_label(0.85) == "strong"
    assert fit_label(0.70) == "good"
    assert fit_label(0.55) == "moderate"
    assert fit_label(0.40) == "weak"
    assert fit_label(0.10) == "poor"


def test_forward_return_uses_one_bar_execution_lag():
    prices = pd.DataFrame({"AAA": [100.0, 110.0, 121.0]})
    # decision_idx=1 -> entry at idx 1 (110), exit at idx 2 (121)
    assert forward_return(prices, decision_idx=1, ticker="AAA", horizon_days=1) == pytest.approx(0.10)


def test_forward_return_none_when_out_of_range():
    prices = pd.DataFrame({"AAA": [100.0, 110.0]})
    assert forward_return(prices, decision_idx=1, ticker="AAA", horizon_days=5) is None


def test_max_drawdown_basic():
    assert max_drawdown([1.0, 1.2, 0.9, 1.1]) == pytest.approx(0.9 / 1.2 - 1.0)
    assert max_drawdown([]) == 0.0


def test_walk_forward_split_keeps_both_sides_non_empty():
    assert compute_walk_forward_split_index(10) == 7
    assert compute_walk_forward_split_index(2) == 1
    assert compute_walk_forward_split_index(1) == 1
    assert compute_walk_forward_split_index(0) == 0


def test_score_combo_rewards_hit_rate_and_shadow_performance():
    base = {
        "activationRate": 0.5,
        "hitRate1D": 0.5,
        "avgForwardReturn1D": 0.0,
        "returnLift1D": 0.0,
        "shadowTotalReturn": 0.0,
        "shadowSharpeLike": 0.0,
        "shadowMaxDrawdown": 0.0,
        "sampleConfidence": 1.0,
    }
    good = {**base, "hitRate1D": 0.8, "avgForwardReturn1D": 0.02, "shadowTotalReturn": 0.3}
    assert score_combo(good, "blended") > score_combo(base, "blended")


def test_score_combo_scales_with_confidence():
    metrics = {
        "activationRate": 0.5, "hitRate1D": 0.8, "avgForwardReturn1D": 0.02,
        "returnLift1D": 0.01, "shadowTotalReturn": 0.3, "shadowSharpeLike": 1.0,
        "shadowMaxDrawdown": -0.05, "sampleConfidence": 1.0,
    }
    low_confidence = {**metrics, "sampleConfidence": 0.1}
    assert score_combo(metrics, "blended") > score_combo(low_confidence, "blended")


def test_calculate_algorithm_fit_empty_returns_poor_zero():
    fit = calculate_algorithm_fit([])
    assert fit["score"] == 0.0
    assert fit["label"] == "poor"
    assert fit["outOfSampleValidated"] is False


def test_calculate_algorithm_fit_returns_raw_and_effective_scores():
    fit = calculate_algorithm_fit([
        {"lookbackDays": 10, "topN": 1, "fitnessScore": 0.9, "sampleConfidence": 0.9, "activationCount": 40, "eligibleDays": 200, "horizonObservationCounts": {"1": 40}},
        {"lookbackDays": 10, "topN": 2, "fitnessScore": 0.85, "sampleConfidence": 0.9, "activationCount": 40, "eligibleDays": 200, "horizonObservationCounts": {"1": 40}},
        {"lookbackDays": 15, "topN": 1, "fitnessScore": 0.88, "sampleConfidence": 0.9, "activationCount": 40, "eligibleDays": 200, "horizonObservationCounts": {"1": 40}},
        {"lookbackDays": 15, "topN": 2, "fitnessScore": 0.92, "sampleConfidence": 0.9, "activationCount": 40, "eligibleDays": 200, "horizonObservationCounts": {"1": 40}},
    ])
    assert fit["rawScore"] >= fit["score"]
    assert fit["confidenceAdjustment"] == "conservative_penalty"
    assert FIT_SCOPE == "universe_conditioned"


def test_calculate_algorithm_fit_prefers_out_of_sample_over_in_sample_winner():
    combo_rows = [
        {"lookbackDays": 10, "topN": 1, "fitnessScore": 0.95, "sampleConfidence": 0.9, "activationCount": 40, "eligibleDays": 200, "horizonObservationCounts": {"1": 40}},
        {"lookbackDays": 15, "topN": 1, "fitnessScore": 0.80, "sampleConfidence": 0.9, "activationCount": 40, "eligibleDays": 200, "horizonObservationCounts": {"1": 40}},
    ]
    # The in-sample winner (lookback=10) performed terribly out-of-sample.
    weak_oos = {"lookbackDays": 10, "topN": 1, "fitnessScore": -0.5, "sampleConfidence": 0.9, "activationCount": 40, "eligibleDays": 200, "horizonObservationCounts": {"1": 40}}
    fit_with_oos = calculate_algorithm_fit(combo_rows, out_of_sample_profile=weak_oos)
    fit_without_oos = calculate_algorithm_fit(combo_rows)
    assert fit_with_oos["calibration"]["outOfSampleComboQuality"] < fit_without_oos["calibration"]["outOfSampleComboQuality"]
    assert fit_with_oos["score"] < fit_without_oos["score"]
