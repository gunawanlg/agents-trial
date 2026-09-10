# scorecard-segment-eval

Sub-population evaluation for logistic scorecards. The package answers two
questions about a business segment carved out of a scored portfolio:

* **Q1 — is the pooled score good enough on this segment?** Discrimination,
  calibration and shape are each judged against explicit gates.
* **Q2 — is a same-predictor refit worth a dedicated model, or is a
  recalibration enough?** A refit only earns a split when it beats the pooled
  score on a forward holdout *and* its predictors are stable over vintages.

Every threshold lives on `Gates`, so a verdict is always traceable to a number
somebody chose rather than to a hidden constant.

## Install

```bash
pip install -e .                # core, runs on Python 3.6+
pip install -e ".[all]"         # plus optbinning, xgboost and matplotlib
pip install -e ".[dev]"         # plus pytest and vermin
pip install -e ".[plot]"        # matplotlib, for grouping vintage plots
```

`optbinning`, `xgboost` and `matplotlib` are strictly optional. When either
accelerator is absent the package warns once and falls back to a
scikit-learn-only path: a decision-tree grouping instead of `optbinning`, and
logistic regression instead of the XGBoost sub-model. Grouping vintage plots
raise a clear `ImportError` until `matplotlib` is installed.

## The two entry paths

### Analysis-ready frame

If you already have a frame with every column resolved, describe it with
`ScorecardColumns` and call `evaluate_segments`:

```python
from scorecard_segment_eval import Gates, ScorecardColumns, evaluate_segments

cols = ScorecardColumns(
    col_id="SKP_CREDIT_CASE",
    col_date="DATE_DECISION",
    col_score="PD",
    col_target="TargetA",
    col_obs="TargetAObs",
    cols_pred=["x1", "x2", "cat"],
    cols_pred_woe=["x1_woe"],
    cols_segment=["CHANNEL"],
)
# grouping is optional: a BinningModel, a parsed SQL scorecard, or omit.
result = evaluate_segments(df, cols, Gates(n_jobs=4), grouping=grouping)
written = result.save_artifacts("artefacts/")
```

Pass `submodel=True` to add an XGBoost sub-model alongside the logistic refit.
`max_depth` is hard-capped at 3 so the sub-model cannot learn beyond a 3-way
interaction; other parameters are overridable through `xgb_params`.

### Smart data creation

More often you only know the table and a handful of columns. Supply the
mandatory inputs and let the resolver infer the rest from the database:

```python
from scorecard_segment_eval import confirm_settings, load_settings, resolve_metadata

meta = resolve_metadata(
    table="risk.base_table",
    col_id="SKP_CREDIT_CASE",
    col_score="PD",
    cols_pred_used=["x1_woe", "x2"],
    executor=executor,
)
outcome = confirm_settings(meta)   # shows the summary, asks, then persists
if outcome.confirmed:
    meta = load_settings(outcome.settings_path)   # lossless on later runs
```

`confirm_settings` prints the resolved metadata and the capability report,
asks for confirmation, and writes the settings on a yes — prompting for the
filename when you do not pass `settings_path`. Tests and scripts bypass the
prompt with `auto_confirm=True`, with `SCORECARD_EVAL_AUTO_CONFIRM=1`, or just
by running without a TTY; all three auto-confirm and warn that they did.

Mandatory: `col_id`, `col_score`, and either `cols_pred_used` or a production
scorecard SQL file (`model_sql` / `model_sql_path`). Inferable from the
database: `col_date`, `cols_segment`, `col_target`, `col_obs`. Every inferred
value raises a `MetadataInferenceWarning` naming the field and how it was
derived, so an inference is never silent.

### Production scorecard SQL

A logistic scorecard written as SQL (CASE WHEN WoE bins, `nvl(LN(p/(1-p)), impute)`
logit / VAL columns, then `LINEAR_SCORE = B^T X` folded through a sigmoid)
is enough to reconstruct the pooled model. `parse_scorecard_sql` /
`parse_scorecard_sql_path` extract:

* `cols_pred` — raw source columns (`indosat_v2`, `featureB`, ...)
* `cols_pred_woe` — `_WOE` aliases, even when the name does not match
  (`indosat_v2` → `feature_a_WOE`)
* `cols_pred_used` — columns in the linear formula (`_WOE`, `_VAL`, or `_LIN`)
* `formula` — `PD = 1/(1+exp(-LINEAR_SCORE))` with `LINEAR_SCORE = B^T X`
* `grouping` — bins plus SQL null imputation, written as `grouping.json`
* `model` — a sklearn `LogisticRegression` with `coef_` / `intercept_` set so
  `predict_proba` matches the SQL sigmoid (the estimator is not fitted)

`render_scorecard_sql` (and `ScorecardSQLModel.to_sql`) write the same query
shape from a grouping and coefficient map, so a refit can ship a
`scorecard.sql` next to `grouping.json`.

```python
from scorecard_segment_eval import (
    evaluate_segments, parse_scorecard_sql_path, resolve_metadata,
)

parsed = parse_scorecard_sql_path("scorecard.sql")
parsed.save_grouping("grouping.json")
print(parsed.formula)
print(parsed.pred_map)          # indosat_v2 -> feature_a_WOE, ...
df["PD"] = parsed.predict_proba(df)[:, 1]

meta = resolve_metadata(
    table="risk.base_table",
    col_id="SKP_CREDIT_CASE",
    col_score="PD",
    model_sql_path="scorecard.sql",
    grouping_path="grouping.json",   # written if the file does not exist
    executor=executor,
)
result = evaluate_segments(df, meta.columns, grouping=meta.grouping)
# grouping=parsed is accepted too: evaluate_segments unwraps .grouping
```

SQL `WHEN x < t` / `x >= t` chains are left-closed `[a, b)` bins. Null and else
branches are stored as `impute` / `missing_note` on each feature in
`grouping.json`. Mixed CASE expressions (string equals then numeric cuts)
keep first-match SQL semantics.

`resolve_capabilities` then reports which analyses are available and which are
blocked. Without `cols_pred` (and with no scorecard SQL to infer it from)
there is no refit; without `cols_pred_woe` or a grouping there is no
recalibration and no grouping-based PSI.

`notebooks/demo_segment_eval.ipynb` runs this flow end to end against a fake
executor, so it needs no database. It also walks the SQL parser, the
`cols_pred` → `cols_pred_woe` map, the segment-vs-portfolio grouping comparison,
and the saved refit / recalibration artefacts.

`notebooks/q2_actions.ipynb` is the shorter walk-through of the three Q2
actions on the synthetic book: `KEEP_POOLED` (core), `RECALIBRATE` (miscal)
and `SPLIT` (inverted), including the grouping vintage plot.

## Conventions worth knowing

### SQL lives in files, never in Python

Every statement is a parameterised template under
`scorecard_segment_eval/sql/`, rendered by `render_sql`. To add a query, drop a
`.sql` file in that directory and use `{placeholder}` for parameters;
`available_queries` picks it up by filename with no Python change.

Rendered SQL carries two machine-readable comment lines:

```sql
-- query: fetch_portfolio
-- params: {"table": "risk.base", "col_id": "SKP_CREDIT_CASE"}
```

A real database ignores comments. `FakeSqlExecutor` reads them instead of
parsing SQL, which is what lets the tests and the notebook exercise the real
rendering path without shipping a SQL parser.

Database access is any callable taking a SQL string and returning a
`DataFrame`. That is the entire executor contract, so a production connection,
a caching wrapper and `FakeSqlExecutor` are interchangeable.

### Target inference is a mapping you extend

`config.py` holds the constants an installation is likely to customise.
`PORTFOLIO_TARGET_MAP` maps a portfolio to its target and observation flag
(`PortfolioA` implies `TargetA` and `TargetAObs`); add a row to teach the
resolver about a new portfolio. Failing that, a supplied `col_obs` implies its
`col_target` and vice versa via `OBS_SUFFIX`, and the last resort is
`TargetDefault` / `TargetDefaultObs`. The column-name candidate tuples used to
sniff the table catalogue for dates, portfolios and segments live here too.

### Mapping `cols_pred` onto `cols_pred_woe`

`map_pred_to_woe` pairs raw predictors with the WoE columns the pooled model
consumes. The usual convention is a `_woe` suffix; when that exact name is
absent it still matches a unique remaining column by prefix, case, or a
`woe_` affix. Each WoE column is used at most once:

```python
from scorecard_segment_eval import map_pred_to_woe

map_pred_to_woe(["predA", "predB"], ["predA_woe"])
# {'predA': 'predA_woe'}
```

`ScorecardColumns.pred_woe_map()` applies the same heuristic, and an explicit
`pred_map` (from parsed scorecard SQL) wins for renamed aliases such as
`indosat_v2` → `feature_a_WOE`. The resolved map is printed in the metadata
summary and is what recalibration and grouping reconstruction use.

`map_pred_to_val` is the VAL / LIN counterpart (`featE` → `featE_VAL`). When a
predictor has both a WoE alias and a `_VAL` / `_LIN` alias, a same-predictor
refit **keeps the logit form** `log(p/(1-p))` instead of re-binning it as WoE.
`ScorecardColumns.logit_pred_cols(grouping)` is the list `evaluate_segments`
passes through; an explicit VAL/LIN alias beats a WoE alias, then a logit
spec on the portfolio / SQL grouping, then a raw column that already looks like
`_VAL` / `_LIN`. A caller-supplied grouping is never rewritten.

### Grouping is serialisable

`BinningModel` fits optimal WoE bins per predictor and round-trips through
`save_grouping` / `load_grouping` as `grouping.json`. Passing a grouping into
`evaluate_segments` switches PSI onto those bins; numeric features without a
spec fall back to portfolio-level decile edges. `psi_method` in the
characteristics table records which path each feature took.

`grouping.vintage_stability_table(...)` is the long frame of per-bin true
event rate, share and univariate Gini over vintages.
`grouping.plot_vintage_stability(...)` draws those as one row of three subplots
(requires `matplotlib`), with a date-formatted x-axis and `__missing__` kept
in the legend even when a vintage has no missings. Adjacent bins whose vintage event-rate Wilson
intervals overlap are merged (`merge_overlapping_event_rate_bounds`);
remaining overlaps are flagged as `overlapping_event_rate_bounds` on the
stability table.

A supplied grouping is also the **portfolio baseline**. A refit fits new
segment WoE bins for ordinary predictors, but keeps SQL `_VAL` / `_LIN`
predictors in logit form. When the grouping itself comes from parsed
scorecard SQL, the segment copy is cloned, its per-bin stats are refreshed on
the training rows, and `compare_groupings` still runs against the original SQL
definition (including mixed CASE features, which are compared on their numeric
edges). `compare_groupings` then notes per-bin edge shifts, merges and
splits, WoE shifts and sign flips. The table is
`result.grouping_comparison`; significant notes are copied onto the segment
`BinSpec` so the saved grouping carries the commentary.

### Fitted model artefacts

Refit and recalibration both return a `FittedModelArtifact` (grouping +
estimator) so later scores can be reproduced. After `evaluate_segments`:

```python
written = result.save_artifacts("artefacts/")
# artefacts/CHANNEL/inverted/refit/{grouping.json, model.pkl, meta.json, scorecard.sql}

from scorecard_segment_eval import load_fitted_artifact
art = load_fitted_artifact(written[0])
pd_hat = art.predict_proba(new_frame)
```

A production SQL scorecard becomes the same kind of artefact via
`parsed.to_artifact()` (`kind="pooled"`, `method="logistic_sql"`). Logistic
refits also write `scorecard.sql` in the same CASE WHEN + `LINEAR_SCORE`
shape as `tests/fixtures/sample_scorecard.sql`, generated by
`render_scorecard_sql`.

### Parallelism is deterministic

`n_jobs` on `Gates` controls per-segment evaluation, bootstrap replicates,
per-predictor binning and stability, and per-characteristic PSI.
`n_jobs=1` runs serially; `n_jobs=-1` uses the core count capped by
`max_workers_cap`. RNG-dependent results are seeded per work item rather than
from shared global state, so output is identical at any worker count — a test
asserts serial and parallel results match exactly. Nested parallelism is
suppressed, so an inner bootstrap inside an already-parallel segment does not
multiply threads.

## Python 3.6 compatibility

3.6 support is a hard requirement, so the package avoids PEP 604 and PEP 585
annotations, the walrus operator, f-string `=`, positional-only parameters,
`functools.cached_property`, `dict |` and `math.prod`. On 3.6, `dataclasses`
comes from the backport, pulled in by a conditional install requirement.

The constraint is checked mechanically rather than by eye:

```bash
vermin -t=3.6- --violations --eval-annotations \
  --backport dataclasses --backport typing scorecard_segment_eval tests
python tests/py36_ast_check.py scorecard_segment_eval tests
```

`tests/py36_ast_check.py` covers the two constructs vermin 1.8 does not flag,
PEP 604 unions in annotations and `from __future__ import annotations`. Both
checks run as part of `tests/test_py36_compat.py`.

## Reporting

`decision_table` and `action_list` give the compact tabular verdicts.
`render_html_report` and `render_markdown_report` build a self-contained
report — no template engine, no new dependency — with per-segment verdicts, the
matched-approval-rate Gini comparison, the PSI method per characteristic, the
segment-vs-portfolio grouping comparison, refit performance and stability
findings, and a prioritised recommendation list from `recommendations`.
`save_report` picks the format from the file extension.

## Tests

```bash
python -m pytest
```
