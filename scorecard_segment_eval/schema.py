from dataclasses import dataclass, field

from typing import List, Optional


@dataclass(frozen=True)
class ScorecardColumns:
    col_id: str
    col_date: str
    col_obs: str
    col_target: str
    col_score: str
    cols_pred: List[str]
    cols_segment: List[str]
    col_fantomas: Optional[str] = None
    cols_pred_woe: List[str] = field(default_factory=list)
    cols_pred_used: List[str] = field(default_factory=list)
    grouping_path: Optional[str] = None
    score_higher_is_risk: bool = True


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
    # Approval-rate alignment for sub-segment Gini
    ar_gap_material: float = 0.10
    n_psi_deciles: int = 10
    # Parallelism: 1 = serial, -1 = all CPUs
    n_jobs: int = -1
    # WoE binning: tree | quantile | optbinning
    woe_strategy: str = "tree"
    woe_tree_max_leaf: int = 8
    woe_tree_min_bin: int = 50
    woe_tree_max_depth: int = 4
    # Optional XGBoost pillar model (max_depth<=3 => at most 3-way interactions)
    submodel: bool = False
    xgb_max_depth: int = 3
    xgb_n_estimators: int = 80
    xgb_learning_rate: float = 0.08
    # Vintage stability gates for refit predictor selection
    stability_psi_max: float = 0.25
    stability_gini_cv_max: float = 0.50
    stability_min_vintages: int = 3
    grouping_path: Optional[str] = None
