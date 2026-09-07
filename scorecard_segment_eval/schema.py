"""Column metadata and gate thresholds.

Python 3.6 compatibility: annotations use ``typing`` generics only, and the
``dataclasses`` backport is declared as a conditional dependency for 3.6.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ScorecardColumns:
    """Physical column names of the analysis table.

    Only ``col_id`` is structurally required.  Everything else may be left
    unset and resolved later by :mod:`scorecard_segment_eval.smartdata`, which
    infers what it can from the database and warns about the rest.
    """

    col_id: str
    col_date: Optional[str] = None
    col_obs: Optional[str] = None
    col_target: Optional[str] = None
    col_score: Optional[str] = None
    cols_pred: List[str] = field(default_factory=list)
    cols_segment: List[str] = field(default_factory=list)
    col_fantomas: Optional[str] = None
    cols_pred_woe: List[str] = field(default_factory=list)
    cols_pred_used: List[str] = field(default_factory=list)

    def as_dict(self):
        # type: () -> Dict[str, object]
        return {
            "col_id": self.col_id,
            "col_date": self.col_date,
            "col_obs": self.col_obs,
            "col_target": self.col_target,
            "col_score": self.col_score,
            "cols_pred": list(self.cols_pred),
            "cols_segment": list(self.cols_segment),
            "col_fantomas": self.col_fantomas,
            "cols_pred_woe": list(self.cols_pred_woe),
            "cols_pred_used": list(self.cols_pred_used),
        }

    @classmethod
    def from_dict(cls, payload):
        # type: (Dict[str, object]) -> "ScorecardColumns"
        known = {
            "col_id",
            "col_date",
            "col_obs",
            "col_target",
            "col_score",
            "cols_pred",
            "cols_segment",
            "col_fantomas",
            "cols_pred_woe",
            "cols_pred_used",
        }
        kwargs = {}
        for key, value in payload.items():
            if key not in known:
                continue
            if key.startswith("cols_") and value is None:
                value = []
            kwargs[key] = value
        return cls(**kwargs)  # type: ignore[arg-type]


@dataclass(frozen=True)
class Gates:
    """Thresholds for the Q1 verdict, the Q2 action and the new diagnostics.

    Existing field names and defaults are preserved; every new capability adds
    fields rather than repurposing old ones.
    """

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

    # --- A1: optimal binning ------------------------------------------------
    binning_method: str = "auto"
    binning_min_bin_frac: float = 0.05
    binning_min_bin_events: int = 10
    binning_max_bins: int = 8
    binning_monotonic: bool = True
    binning_monotonic_iv_tolerance: float = 0.20

    # --- A1: optional XGBoost sub-model ------------------------------------
    submodel_max_depth: int = 3

    # --- A1: predictor stability over vintages ------------------------------
    stability_psi_max: float = 0.25
    stability_gini_ratio_min: float = 0.50
    stability_sign_consistency_min: float = 0.80
    stability_score_min: float = 0.60
    stability_min_vintages: int = 3
    stability_max_weak_vintage_share: float = 0.34
    stability_min_reference_gini: float = 0.05
    refit_requires_stability: bool = True

    # --- A2: matched approval-rate Gini -------------------------------------
    reference_approval_rate: float = 0.85
    ar_gap_trigger: float = 0.10
    matched_ar_min_n: int = 100
    matched_ar_min_events: int = 10

    # --- D: parallelism -----------------------------------------------------
    n_jobs: int = 1
    max_workers_cap: int = 8


#: Analyses that :func:`scorecard_segment_eval.smartdata.resolve_capabilities`
#: reports on.  Kept here so callers can introspect the full universe.
ANALYSIS_KEYS = (
    "segment_performance",
    "vintage_stability",
    "matched_ar_gini",
    "psi_numeric",
    "psi_grouping",
    "recalibration",
    "refit",
    "submodel",
    "report",
)
