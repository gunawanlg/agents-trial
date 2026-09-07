import numpy as np
import pandas as pd

from scorecard_segment_eval.schema import ScorecardColumns


def make_synthetic_book(n=12000, seed=7):
    """Book with four segments: good, miscalibrated, inverted (split), and tiny."""
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 540, size=n), unit="D")
    channel = np.array(["core"] * n, dtype=object)
    n_mis = int(n * 0.18)
    n_inv = int(n * 0.22)
    n_tiny = 80
    channel[:n_mis] = "miscal"
    channel[n_mis : n_mis + n_inv] = "inverted"
    channel[n_mis + n_inv : n_mis + n_inv + n_tiny] = "tiny"

    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    cat = rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])
    cat_eff = np.where(cat == "A", 0.0, np.where(cat == "B", 0.4, -0.3))

    logit = -2.2 + 0.9 * x1 + 0.7 * x2 + cat_eff
    inv = channel == "inverted"
    logit = logit.copy()
    logit[inv] = -2.0 - 1.7 * x1[inv]
    p_true = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, p_true)

    # Pooled score: fitted as if the inverted sign were the majority sign (x1 positive).
    logit_model = -2.2 + 0.9 * x1 + 0.7 * x2 + cat_eff
    p_model = 1 / (1 + np.exp(-logit_model))
    p_model = p_model.copy()
    p_model[channel == "miscal"] = np.clip(p_model[channel == "miscal"] * 2.4, 1e-4, 0.95)

    obs = rng.random(n) < 0.85
    obs[y == 1] = True
    fantomas = (p_model < 0.03) & (rng.random(n) < 0.04)

    df = pd.DataFrame(
        {
            "app_id": np.arange(n),
            "score_date": dates,
            "obs": obs.astype(int),
            "default": y.astype(int),
            "pd": p_model,
            "x1": x1,
            "x2": x2,
            "cat": cat,
            "x1_woe": x1,
            "fantomas": fantomas.astype(int),
            "channel": channel,
        }
    )
    cols = ScorecardColumns(
        col_id="app_id",
        col_date="score_date",
        col_obs="obs",
        col_target="default",
        col_score="pd",
        cols_pred=["x1", "x2", "cat"],
        cols_pred_woe=["x1_woe"],
        cols_pred_used=["x1_woe", "x2", "cat"],
        col_fantomas="fantomas",
        cols_segment=["channel"],
    )
    return df, cols
