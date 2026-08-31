from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ScorecardColumns:
    col_id: str
    col_date: str
    col_obs: str
    col_target: str
    col_score: str
    cols_pred: list[str]
    cols_segment: list[str]
    col_fantomas: str | None = None
    cols_pred_woe: list[str] = field(default_factory=list)
    cols_pred_used: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Gates:
    min_n: int = 1000
    min_events: int = 50
    gini_floor: float = 0.25
    gini_ratio_floor: float = 0.80
    oe_lo: float = 0.80
    oe_hi: float = 1.25
    ece_max: float = 0.03
    vintage_gini_ratio_floor: float = 0.70
    holdout_frac: float = 0.25
    min_delta_gini: float = 0.03
    delta_gini_z: float = 1.64
    importance_share_floor: float = 0.05
    psi_woe_material: float = 0.25
    woe_spearman_reversal: float = 0.0
    n_bootstrap: int = 200
    n_ece_bins: int = 10
    n_woe_bins: int = 10
    bootstrap_seed: int = 42
