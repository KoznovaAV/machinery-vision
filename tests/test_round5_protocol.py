"""Protect source independence and prevent gallery-size inflation of metrics."""

import pytest

from scripts.prepare_round5_dataset import split_sources
from scripts.train_round5_recognition import summarize


def test_same_source_never_crosses_split_even_with_multiple_models():
    rows = [
        dict(key=str(i), model=label, group=group)
        for i, (label, group) in enumerate(
            [("a", "same"), ("b", "same"), ("a", "one"), ("b", "two"), ("a", "three")]
        )
    ]
    result = split_sources(rows)
    for group in {r["group"] for r in result}:
        assert len({r["role"] for r in result if r["group"] == group}) == 1


def test_splits_do_not_depend_on_photo_order():
    rows = [dict(key=str(i), model="nh_t5000", group="source-" + str(i // 3)) for i in range(30)]
    first = {r["key"]: r["role"] for r in split_sources(rows)}
    second = {r["key"]: r["role"] for r in split_sources(list(reversed(rows)))}
    assert first == second
    assert "test" in first.values() and "train" in first.values()


def test_single_source_does_not_fabricate_independent_control():
    result = split_sources(
        [dict(key=str(i), group="one-machine", model="mtz_820") for i in range(15)]
    )
    assert {r["role"] for r in result} == {"train"}


def test_group_metric_does_not_count_nine_views_as_nine_machines():
    records = [dict(group="easy", cohort="new", expected="a", prediction="a") for _ in range(9)]
    records.append(dict(group="hard", cohort="new", expected="a", prediction="b"))
    result = summarize(records)["new"]
    assert (result["correct"], result["total"], result["groups"]) == (9, 10, 2)
    assert result["group_mean_accuracy"] == pytest.approx(0.5)


@pytest.mark.parametrize("mode", ["fused", "fused_full_tractor"])
def test_new_pilot_feature_modes_keep_manual_only_policy(mode):
    from scripts.predict_field_pilot import validate_pilot_head

    head = dict(
        feature_encoder=mode, decision_policy="manual_review_only", detector_size=[320, 544]
    )
    assert validate_pilot_head(head) == mode
    head["decision_policy"] = "automatic"
    with pytest.raises(ValueError):
        validate_pilot_head(head)


def test_crop_pilot_rejects_changed_detector_preprocessing():
    from scripts.predict_field_pilot import validate_pilot_head

    with pytest.raises(ValueError):
        validate_pilot_head(
            dict(
                feature_encoder="fused_full_tractor",
                decision_policy="manual_review_only",
                detector_size=[480, 800],
            )
        )


def test_unknown_control_model_is_scored_as_other_for_head_scope():
    from scripts.train_round5_recognition import evaluation_label

    assert evaluation_label("nh_td5050", {"nh_t5000", "other"}) == "other"
    assert evaluation_label("acros_550", {"nh_t5000", "other"}) == "other"
    assert evaluation_label("nh_t5000", {"nh_t5000", "other"}) == "nh_t5000"
