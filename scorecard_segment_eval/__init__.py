"""Sub-population evaluation for logistic scorecards.

Q1: is the pooled score good on each business segment?
Q2: is a same-predictor refit worth a split, or is recalibration enough?

The package is written to run on Python 3.6 and up: annotations use
``typing`` generics in comment form, and on 3.6 ``dataclasses`` is pulled in
from the backport by a conditional install requirement.
"""

from scorecard_segment_eval.binning import (
    BinningModel,
    BinSpec,
    compare_groupings,
    fit_bin_spec,
    grouping_from_woe_columns,
    grouping_from_dict,
    load_grouping,
    save_grouping,
)
from scorecard_segment_eval.characteristics import (
    feature_diagnostics,
    psi_for_feature,
    psi_table,
    shape_divergence,
)
from scorecard_segment_eval.dbio import (
    FakeSqlExecutor,
    SqlExecutor,
    available_queries,
    render_sql,
)
from scorecard_segment_eval.decision import q1_verdict, q2_action
from scorecard_segment_eval.evaluate import SegmentEvalResult, evaluate_segments
from scorecard_segment_eval.metrics import matched_ar_comparison, performance_bundle
from scorecard_segment_eval.parallel import map_jobs, resolve_n_jobs
from scorecard_segment_eval.refit import (
    FittedModelArtifact,
    RecalibrateResult,
    RefitResult,
    WoEEncoder,
    load_fitted_artifact,
    recalibrate_with_diagnostics,
    refit_same_predictors,
    refit_with_diagnostics,
    save_fitted_artifact,
    save_fitted_artifacts,
)
from scorecard_segment_eval.report import (
    action_list,
    decision_table,
    recommendations,
    render_html_report,
    render_markdown_report,
    save_report,
)
from scorecard_segment_eval.schema import Gates, ScorecardColumns, map_pred_to_val, map_pred_to_woe
from scorecard_segment_eval.sql_model import (
    ScorecardSQLModel,
    parse_scorecard_sql,
    parse_scorecard_sql_path,
)
from scorecard_segment_eval.smartdata import (
    CapabilityReport,
    ConfirmationResult,
    MetadataInferenceWarning,
    ResolvedMetadata,
    build_analysis_frame,
    confirm_settings,
    load_settings,
    render_metadata_summary,
    resolve_capabilities,
    resolve_metadata,
    save_settings,
)
from scorecard_segment_eval.stability import predictor_stability, stability_summary
from scorecard_segment_eval.synthetic import (
    make_fake_executor,
    make_synthetic_book,
    make_synthetic_db_table,
)

__all__ = [
    # schema / gates
    "Gates",
    "ScorecardColumns",
    "map_pred_to_woe",
    "map_pred_to_val",
    "ScorecardSQLModel",
    "parse_scorecard_sql",
    "parse_scorecard_sql_path",
    # evaluation
    "SegmentEvalResult",
    "evaluate_segments",
    "performance_bundle",
    "matched_ar_comparison",
    "q1_verdict",
    "q2_action",
    # binning / grouping
    "BinSpec",
    "BinningModel",
    "fit_bin_spec",
    "load_grouping",
    "save_grouping",
    "compare_groupings",
    "grouping_from_dict",
    "grouping_from_woe_columns",
    # characteristics / PSI
    "feature_diagnostics",
    "psi_for_feature",
    "psi_table",
    "shape_divergence",
    # refit / stability
    "RefitResult",
    "RecalibrateResult",
    "FittedModelArtifact",
    "WoEEncoder",
    "refit_same_predictors",
    "refit_with_diagnostics",
    "recalibrate_with_diagnostics",
    "save_fitted_artifact",
    "save_fitted_artifacts",
    "load_fitted_artifact",
    "predictor_stability",
    "stability_summary",
    # smart data
    "CapabilityReport",
    "ConfirmationResult",
    "MetadataInferenceWarning",
    "ResolvedMetadata",
    "build_analysis_frame",
    "confirm_settings",
    "load_settings",
    "render_metadata_summary",
    "resolve_capabilities",
    "resolve_metadata",
    "save_settings",
    # database access
    "FakeSqlExecutor",
    "SqlExecutor",
    "available_queries",
    "render_sql",
    # reporting
    "action_list",
    "decision_table",
    "recommendations",
    "render_html_report",
    "render_markdown_report",
    "save_report",
    # parallelism
    "map_jobs",
    "resolve_n_jobs",
    # synthetic data
    "make_fake_executor",
    "make_synthetic_book",
    "make_synthetic_db_table",
]
