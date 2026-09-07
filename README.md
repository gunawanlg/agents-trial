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
pip install -e ".[all]"         # plus optbinning and xgboost
pip install -e ".[dev]"         # plus pytest and vermin
```

`optbinning` and `xgboost` are strictly optional. When either is absent the
package warns once and falls back to a scikit-learn-only path: a decision-tree
grouping instead of `optbinning`, and logistic regression instead of the
XGBoost sub-model.

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
    cols_segment=["CHANNEL"],
)
result = evaluate_segments(df, cols, Gates(n_jobs=4))
```

Pass `submodel=True` to add an XGBoost sub-model alongside the logistic refit.
`max_depth` is hard-capped at 3 so the sub-model cannot learn beyond a 3-way
interaction; other parameters are overridable through `xgb_params`.

### Smart data creation

More often you only know the table and a handful of columns. Supply the
mandatory three and let the resolver infer the rest from the database:

```python
from scorecard_segment_eval import (
    ScorecardColumns, confirm_settings, load_settings, resolve_metadata,
)

meta = resolve_metadata(
    table="risk.base_table",
    cols=ScorecardColumns(col_id="SKP_CREDIT_CASE", col_score="PD",
                          cols_pred_used=["x1", "x2"]),
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

Mandatory: `col_id`, `col_score`, `cols_pred_used`. Inferable from the
database: `col_date`, `cols_segment`, `col_target`, `col_obs`. Every inferred
value raises a `MetadataInferenceWarning` naming the field and how it was
derived, so an inference is never silent.

`resolve_capabilities` then reports which analyses are available and which are
blocked. Two inputs cannot be inferred and only degrade the analysis set:
without `cols_pred` there is no refit, and without `cols_pred_woe` or a
grouping there is no recalibration and no grouping-based PSI.

`notebooks/demo_segment_eval.ipynb` runs this flow end to end against a fake
executor, so it needs no database.

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

### Grouping is serialisable

`BinningModel` fits optimal WoE bins per predictor and round-trips through
`save_grouping` / `load_grouping` as `grouping.json`. Passing a grouping into
`evaluate_segments` switches PSI onto those bins; numeric features without a
spec fall back to portfolio-level decile edges. `psi_method` in the
characteristics table records which path each feature took.

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
matched-approval-rate Gini comparison, the PSI method per characteristic, refit
performance and stability findings, and a prioritised recommendation list from
`recommendations`. `save_report` picks the format from the file extension.

## Tests

```bash
python -m pytest
```
