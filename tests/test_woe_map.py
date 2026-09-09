from scorecard_segment_eval.schema import ScorecardColumns, map_pred_to_val, map_pred_to_woe


def test_suffix_mapping_when_only_some_woe_columns_are_supplied():
    mapping = map_pred_to_woe(["predA", "predB"], ["predA_woe"])
    assert mapping == {"predA": "predA_woe"}


def test_exact_match_when_the_predictor_is_already_the_woe_column():
    mapping = map_pred_to_woe(["x1_woe", "x2"], ["x1_woe", "x2_woe"])
    assert mapping["x1_woe"] == "x1_woe"
    assert mapping["x2"] == "x2_woe"


def test_case_insensitive_and_prefix_affixes():
    mapping = map_pred_to_woe(["Age", "income"], ["WOE_income", "age_WOE"])
    assert mapping["Age"] == "age_WOE"
    assert mapping["income"] == "WOE_income"


def test_unique_prefix_match_when_the_conventional_suffix_is_absent():
    mapping = map_pred_to_woe(["predA", "predB"], ["predA_binned_woe"])
    assert mapping == {"predA": "predA_binned_woe"}


def test_each_woe_column_is_used_at_most_once():
    mapping = map_pred_to_woe(["pred", "pred_extra"], ["pred_woe"])
    assert mapping == {"pred": "pred_woe"}
    assert "pred_extra" not in mapping


def test_empty_inputs_yield_an_empty_map():
    assert map_pred_to_woe(None, ["x_woe"]) == {}
    assert map_pred_to_woe(["x"], None) == {}
    assert map_pred_to_woe([], ["x_woe"]) == {}


def test_scorecard_columns_exposes_the_same_map():
    cols = ScorecardColumns(
        col_id="id",
        cols_pred=["predA", "predB"],
        cols_pred_woe=["predA_woe"],
    )
    assert cols.pred_woe_map() == {"predA": "predA_woe"}


def test_ambiguous_prefix_is_not_guessed():
    mapping = map_pred_to_woe(["pred"], ["pred_woe", "pred_other_woe"])
    # Conventional suffix wins over the extra column.
    assert mapping == {"pred": "pred_woe"}


def test_map_pred_to_val_suffix_and_lin():
    mapping = map_pred_to_val(["featE", "featF", "x"], ["featE_VAL", "featF_LIN"])
    assert mapping == {"featE": "featE_VAL", "featF": "featF_LIN"}
    assert "x" not in mapping


def test_map_pred_to_val_case_insensitive():
    mapping = map_pred_to_val(["featE"], ["feate_val"])
    assert mapping["featE"] == "feate_val"


def test_scorecard_columns_val_map_and_logit_priority():
    cols = ScorecardColumns(
        col_id="id",
        cols_pred=["featE", "featureB"],
        cols_pred_woe=["featE_WOE", "featureB_WOE"],
        cols_pred_used=["featE_WOE", "featE_VAL", "featureB_WOE"],
        pred_map={"featE": "featE_WOE", "featureB": "featureB_WOE"},
    )
    assert cols.pred_woe_map()["featE"] == "featE_WOE"
    assert cols.pred_val_map()["featE"] == "featE_VAL"
    assert "featureB" not in cols.pred_val_map()
    assert cols.logit_pred_cols() == ["featE"]


def test_logit_pred_cols_from_grouping_spec():
    from scorecard_segment_eval.binning import BinSpec, BinningModel

    grouping = BinningModel(
        specs={
            "pd_raw": BinSpec(
                feature="pd_raw",
                kind="logit",
                method="sql_logit",
                transform="logit",
                labels=["logit"],
            )
        }
    )
    cols = ScorecardColumns(col_id="id", cols_pred=["pd_raw", "x1"])
    assert cols.logit_pred_cols(grouping=grouping) == ["pd_raw"]
