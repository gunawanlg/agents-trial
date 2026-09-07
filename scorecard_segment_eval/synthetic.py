"""Synthetic books for tests, demos and the notebook.

:func:`make_synthetic_book` returns an analysis-ready frame (the historical
behaviour).  :func:`make_synthetic_db_table` returns the *same* book shaped
like a production database table -- production column names, a portfolio
column, no analysis-ready assumptions -- so the smart-data metadata resolution
can be demonstrated end to end without a real database.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from scorecard_segment_eval.schema import ScorecardColumns

#: Mapping from analysis-ready names to the production-style names used by
#: :func:`make_synthetic_db_table`.
DB_COLUMN_NAMES = {
    "app_id": "SKP_CREDIT_CASE",
    "score_date": "DATE_DECISION",
    "obs": "TargetAObs",
    "default": "TargetA",
    "pd": "PD",
    "channel": "CHANNEL",
    "fantomas": "FLAG_FANTOMAS",
}


def make_synthetic_book(n=12000, seed=7):
    # type: (int, int) -> Tuple[pd.DataFrame, ScorecardColumns]
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


def make_synthetic_db_table(n=12000, seed=7, portfolio="PortfolioA", rename=None):
    # type: (int, int, Optional[str], Optional[Dict[str, str]]) -> pd.DataFrame
    """The synthetic book with production-style column names and a portfolio.

    Feed this to :class:`scorecard_segment_eval.dbio.FakeSqlExecutor` to
    exercise metadata inference: only ``SKP_CREDIT_CASE``, ``PD`` and the
    predictor list need to be supplied, and the date, segmentation, target and
    observation flag are all discoverable from the table itself.
    """
    df, _cols = make_synthetic_book(n=n, seed=seed)
    mapping = dict(DB_COLUMN_NAMES)
    mapping.update(rename or {})
    table = df.rename(columns=mapping)
    if portfolio is not None:
        table["PORTFOLIO"] = str(portfolio)
    ordered = [c for c in ("SKP_CREDIT_CASE", "DATE_DECISION", "PORTFOLIO") if c in table.columns]
    ordered.extend(c for c in table.columns if c not in ordered)
    return table.loc[:, ordered]


def make_fake_executor(table=None, n=12000, seed=7, portfolio="PortfolioA", hidden_columns=()):
    # type: (Optional[pd.DataFrame], int, int, Optional[str], Sequence[str]) -> object
    """A :class:`~scorecard_segment_eval.dbio.FakeSqlExecutor` over the book."""
    from scorecard_segment_eval.dbio import FakeSqlExecutor

    frame = table if table is not None else make_synthetic_db_table(n=n, seed=seed, portfolio=portfolio)
    return FakeSqlExecutor(table=frame, hidden_columns=hidden_columns)
