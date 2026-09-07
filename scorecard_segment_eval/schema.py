from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ScorecardColumns:
    col_id: str = "SKP_CREDIT_CASE"
    col_date: str = "date"
    col_obs: str = "TargetDefaultObs"
    col_target: str = "TargetDefault"
    col_score: str = ""
    cols_pred: List[str] = field(default_factory=list)
    cols_segment: List[str] = field(default_factory=list)
    col_fantomas: Optional[str] = None
    cols_pred_woe: List[str] = field(default_factory=list)
    cols_pred_used: List[str] = field(default_factory=list)
    grouping: Dict[str, Any] = field(default_factory=dict)


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
    n_psi_bins: int = 10
    approval_rate_gap: float = 0.10
    refit_stability_penalty: float = 0.05
    max_woe_psi: float = 0.25
    bootstrap_seed: int = 42
