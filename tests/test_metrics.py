import numpy as np

from scorecard_segment_eval.metrics import brier, gini, ks_stat, observed_expected
from scorecard_segment_eval.population import time_holdout_mask
import pandas as pd


def test_gini_perfect_ranking():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    assert gini(y, p) == 1.0


def test_ks_perfect_separation():
    y = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95])
    assert ks_stat(y, p) == 1.0


def test_oe_and_brier():
    y = np.array([0.0, 0.0, 1.0, 1.0])
    p = np.array([0.25, 0.25, 0.75, 0.75])
    oe = observed_expected(y, p)
    assert abs(oe["oe"] - 1.0) < 1e-9
    assert brier(y, p) == 0.0625


def test_time_holdout_uses_latest_dates():
    dates = pd.Series(pd.date_range("2024-01-01", periods=10, freq="D"))
    mask = time_holdout_mask(dates, holdout_frac=0.3)
    assert mask.sum() >= 1
    assert dates[mask].min() > dates[~mask].max()
