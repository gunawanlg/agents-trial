"""Column metadata and gate thresholds.

Python 3.6 compatibility: annotations use ``typing`` generics only, and the
``dataclasses`` backport is declared as a conditional dependency for 3.6.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


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

    def pred_woe_map(self):
        # type: () -> Dict[str, str]
        """Map each raw predictor onto its supplied WoE column, when one exists."""
        return map_pred_to_woe(self.cols_pred, self.cols_pred_woe)


#: Affixes used to recognise a Weight-of-Evidence column name.
_WOE_SUFFIXES = ("_woe", "_woes")
_WOE_PREFIXES = ("woe_", "woe")


def _strip_woe_affix(name):
    # type: (str) -> str
    text = str(name).strip()
    lowered = text.lower()
    for suffix in _WOE_SUFFIXES:
        if lowered.endswith(suffix) and len(lowered) > len(suffix):
            return text[: len(text) - len(suffix)]
    for prefix in _WOE_PREFIXES:
        if lowered.startswith(prefix) and len(lowered) > len(prefix):
            rest = text[len(prefix) :]
            if rest.startswith("_"):
                rest = rest[1:]
            return rest
    return text


def _norm(name):
    # type: (str) -> str
    return str(name).strip().lower()


def map_pred_to_woe(cols_pred, cols_pred_woe):
    # type: (Optional[Sequence[str]], Optional[Sequence[str]]) -> Dict[str, str]
    """Map ``cols_pred`` onto ``cols_pred_woe``.

    The usual convention is a ``_woe`` suffix (``predA`` → ``predA_woe``).
    When that exact name is absent, the mapper still pairs a predictor with a
    unique remaining WoE column by prefix, case-insensitive name, or by
    stripping a ``woe_`` / ``_woe`` affix.  Each WoE column is used at most
    once, in ``cols_pred`` order.  Predictors with no counterpart are omitted.

    Example::

        map_pred_to_woe(['predA', 'predB'], ['predA_woe'])
        # {'predA': 'predA_woe'}
    """
    preds = [str(c) for c in (cols_pred or ()) if str(c)]
    woes = [str(c) for c in (cols_pred_woe or ()) if str(c)]
    if not preds or not woes:
        return {}

    remaining = list(woes)
    mapping = {}  # type: Dict[str, str]

    def _take(candidate):
        # type: (Optional[str]) -> Optional[str]
        if candidate is None or candidate not in remaining:
            return None
        remaining.remove(candidate)
        return candidate

    def _find_ci(target):
        # type: (str) -> Optional[str]
        needle = _norm(target)
        hits = [w for w in remaining if _norm(w) == needle]
        return hits[0] if hits else None

    for pred in preds:
        chosen = None  # type: Optional[str]
        # 1. Exact name: the predictor *is* already the WoE column.
        chosen = _take(pred) if pred in remaining else None
        # 2. Conventional suffix / prefix, preserving the caller's spelling.
        if chosen is None:
            for candidate in (pred + "_woe", pred + "_WoE", pred + "_WOE", "woe_" + pred, "WOE_" + pred):
                chosen = _take(candidate)
                if chosen is not None:
                    break
        # 3. Case-insensitive exact, then conventional affix.
        if chosen is None:
            chosen = _take(_find_ci(pred))
        if chosen is None:
            for candidate in (pred + "_woe", "woe_" + pred):
                chosen = _take(_find_ci(candidate))
                if chosen is not None:
                    break
        # 4. Unique remaining WoE column whose affix-stripped name matches.
        if chosen is None:
            want = _norm(_strip_woe_affix(pred))
            hits = [w for w in remaining if _norm(_strip_woe_affix(w)) == want]
            if len(hits) == 1:
                chosen = _take(hits[0])
        # 5. Unique remaining WoE column that starts with the predictor
        #    (predA → predA_woe even if other affixes were used).
        if chosen is None:
            needle = _norm(pred)
            hits = [w for w in remaining if _norm(w).startswith(needle) and _norm(w) != needle]
            if len(hits) == 1:
                chosen = _take(hits[0])
        # 6. Unique remaining WoE column whose stripped name starts with the
        #    predictor, or vice versa, when that is unambiguous.
        if chosen is None:
            want = _norm(_strip_woe_affix(pred))
            hits = [
                w
                for w in remaining
                if _norm(_strip_woe_affix(w)).startswith(want) or want.startswith(_norm(_strip_woe_affix(w)))
            ]
            if len(hits) == 1:
                chosen = _take(hits[0])
        if chosen is not None:
            mapping[pred] = chosen
    return mapping


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

    # --- A1: segment vs portfolio grouping comparison ------------------------
    grouping_woe_shift_material: float = 0.25
    grouping_edge_shift_frac: float = 0.20
    grouping_sign_flip_floor: float = 0.05

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
