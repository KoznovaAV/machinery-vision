"""Hierarchy contracts: uncertain granularity, local evidence and source calibration."""

import numpy as np
import pytest

from src.training.grouped import calibrate, specialist_rows
from src.config.machinery_taxonomy import TAXONOMY_SHA256, TAXONOMY_VERSION, label_resolution
from src.models.hierarchy import predict_hierarchy, visible_marking, vote
from src.models.machinery_regions import select_regions


def constant(classes, winner):
    return dict(
        classes=classes,
        coef=np.zeros((len(classes), 2)).tolist(),
        intercept=[10 if c == winner else -10 for c in classes],
    )


def bundle(family="mtz_80_family", kind="tractor"):
    return dict(
        taxonomy_version=TAXONOMY_VERSION,
        taxonomy_sha256=TAXONOMY_SHA256,
        decision_policy="hierarchical_review_only",
        heads=dict(
            type=constant(["tractor", "combine_harvester"], kind),
            family=constant(["mtz_80_family", "nh_t5000_family", "unknown"], family),
            mtz_specialist=constant(["mtz_80", "mtz_82", "mtz_820"], "mtz_80"),
        ),
        calibration=dict(
            type={"tractor": {"threshold": 0.8}},
            family={"mtz_80_family": {"threshold": 0.8}, "nh_t5000_family": {"threshold": 0.8}},
        ),
    )


def test_family_only_never_certifies_model_group():
    result = predict_hierarchy(bundle(), np.zeros(2))
    assert result["status"] == "family_only"
    assert result["family"] == "mtz_80_family"
    assert result["model"] is None and result["exact_model"] is None
    assert result["candidate_resolution"] == "model_group"


def test_marking_cannot_promote_series_to_exact_identity():
    evidence = dict(
        status="consistent_marking",
        suggestion="nh_t5000",
        pairs=[dict(full_text="T5050", body_text="T5050")],
    )
    result = predict_hierarchy(bundle("nh_t5000_family"), np.zeros(2), evidence)
    assert result["visible_model_marking"] == "T5050"
    assert result["exact_model"] is None
    assert result["candidate_resolution"] == "series"


def test_conflicting_marking_withholds_family_answer():
    result = predict_hierarchy(
        bundle(), np.zeros(2), dict(status="consistent_marking", suggestion="nh_t5000")
    )
    assert result["status"] == "manual_review" and result["family"] is None
    assert result["marking_conflict"]


def test_disagreeing_type_withholds_model_candidate():
    result = predict_hierarchy(bundle(), np.zeros(2), detected_type="combine_harvester")
    assert result["family"] is None and result["model_candidate"] is None
    assert result["type_conflict"]


def test_untrained_bulldozer_is_not_forced_into_tractor_model():
    result = predict_hierarchy(bundle(), np.zeros(2), detected_type="bulldozer")
    assert result["vehicle_type_candidate"] == "bulldozer"
    assert result["model_candidate"] is None


def test_absent_calibration_means_manual_review():
    b = bundle()
    b["calibration"]["family"]["mtz_80_family"]["threshold"] = None
    assert predict_hierarchy(b, np.zeros(2))["status"] == "manual_review"


def test_binary_sklearn_head_uses_both_classes():
    head = dict(classes=["combine_harvester", "tractor"], coef=[[1.0, 0.0]], intercept=[0.0])
    assert vote(head, np.array([2.0, 0.0]))[0] == "tractor"
    assert vote(head, np.array([-2.0, 0.0]))[0] == "combine_harvester"


def test_nonfinite_features_are_rejected():
    with pytest.raises(ValueError):
        vote(constant(["a", "b"], "a"), np.array([np.nan, 0.0]))


def test_calibration_cannot_count_many_photos_of_one_machine_as_independent():
    records = [dict(group="one", expected="a", probabilities=[0.99, 0.01]) for _ in range(30)]
    assert calibrate(records, ["a", "b"])["a"]["threshold"] is None


def test_calibration_uses_independent_groups():
    records = [dict(group=str(i), expected="a", probabilities=[0.9, 0.1]) for i in range(12)]
    result = calibrate(records, ["a", "b"])["a"]
    assert result["groups"] == 12 and result["training_oof_precision"] == 1.0


def test_other_families_do_not_change_specialist_training_membership():
    rows = [
        dict(key=str(i), model=label) for i, label in enumerate(["mtz_80", "mtz_82", "mtz_820"])
    ]
    assert specialist_rows(rows + [dict(key="extra", model="acros_550")]) == specialist_rows(rows)


def test_exact_marking_requires_identical_digits_in_both_views():
    assert visible_marking(dict(pairs=[dict(full_text="T5050", body_text="T5060")])) is None
    assert visible_marking(dict(pairs=[dict(full_text="80.1", body_text="80")])) is None


def test_series_taxonomy_is_explicit():
    assert label_resolution("nh_t5000") == "series"
    assert label_resolution("claas_axion_800") == "series"
    assert label_resolution("mtz_80") == "model_group"


def test_two_separate_machines_survive_nms():
    raw = [
        dict(box=[0, 0, 100, 100], score=0.9, label="tractor"),
        dict(box=[150, 0, 250, 100], score=0.8, label="tractor"),
        dict(box=[1, 1, 99, 99], score=0.7, label="bulldozer"),
    ]
    assert len(select_regions(raw, 300, 120)) == 2


def test_tiny_background_object_does_not_get_model_identity():
    assert not select_regions([dict(box=[0, 0, 10, 10], score=0.9, label="tractor")], 1000, 1000)


def test_changed_taxonomy_cannot_reinterpret_old_weights():
    b = bundle()
    b["taxonomy_sha256"] = "changed"
    with pytest.raises(ValueError):
        predict_hierarchy(b, np.zeros(2))


def test_family_group_is_not_returned_as_a_unique_model_candidate():
    result = predict_hierarchy(bundle(), np.zeros(2))
    assert result["label_candidate"] == "mtz_80"
    assert result["model_candidate"] is None


def test_verified_digits_can_suggest_registered_member_without_confirming_it():
    evidence = dict(
        status="consistent_marking",
        suggestion="nh_t5000",
        pairs=[dict(full_text="T5050", body_text="T5050")],
    )
    result = predict_hierarchy(bundle("nh_t5000_family"), np.zeros(2), evidence)
    assert result["model_candidate"] == "nh_t5050"
    assert result["exact_model"] is None and result["model"] is None


def test_marking_whitespace_does_not_change_model_digits():
    assert visible_marking(dict(pairs=[dict(full_text="T 5050", body_text="T5050")])) == "T5050"


def test_nested_night_tractor_boxes_are_one_machine():
    raw = [
        dict(box=[498, 424, 1011, 796], score=0.8, label="tractor"),
        dict(box=[390, 370, 1122, 849], score=0.7, label="tractor"),
    ]
    assert len(select_regions(raw, 1600, 1000)) == 1


def test_overlapping_machines_with_different_centres_stay_separate():
    raw = [
        dict(box=[10, 10, 210, 210], score=0.8, label="tractor"),
        dict(box=[130, 10, 330, 210], score=0.7, label="tractor"),
    ]
    assert len(select_regions(raw, 400, 250)) == 2


def test_offcentre_foreground_machine_is_not_a_centered_duplicate():
    raw = [
        dict(box=[10, 10, 310, 210], score=0.8, label="tractor"),
        dict(box=[20, 20, 120, 100], score=0.7, label="tractor"),
    ]
    assert len(select_regions(raw, 400, 250)) == 2


def test_optional_unknown_gate_requires_calibrated_evidence():
    b = bundle()
    b["heads"]["unknown_gate"] = constant(["known", "unknown"], "unknown")
    b["calibration"]["unknown_gate"] = {"unknown": {"threshold": None}}
    assert predict_hierarchy(b, np.zeros(2))["family_candidate"] == "mtz_80_family"
    b["calibration"]["unknown_gate"]["unknown"]["threshold"] = 0.95
    prediction = predict_hierarchy(b, np.zeros(2))
    assert prediction["family_candidate"] == "unknown"
    assert prediction["family"] is None


def test_nh_specialist_uses_its_own_calibration_without_certifying_exact_model():
    b = bundle()
    b["heads"]["nh_specialist"] = constant(["nh_t5000_family", "not_nh"], "nh_t5000_family")
    b["calibration"]["nh_specialist"] = {"nh_t5000_family": {"threshold": 0.95}}
    b["calibration"]["family"]["nh_t5000_family"] = {"threshold": 1.0}
    prediction = predict_hierarchy(b, np.zeros(2))
    assert prediction["family"] == "nh_t5000_family"
    assert prediction["exact_model"] is None


def test_uncalibrated_nh_specialist_cannot_change_family():
    b = bundle()
    b["heads"]["nh_specialist"] = constant(["nh_t5000_family", "not_nh"], "nh_t5000_family")
    b["calibration"]["nh_specialist"] = {"nh_t5000_family": {"threshold": None}}
    assert predict_hierarchy(b, np.zeros(2))["family"] == "mtz_80_family"


def branded_evidence(token="82", suggestion="mtz_82", brand="MTZ"):
    return dict(
        status="consistent_marking",
        suggestion=suggestion,
        image_size=[1000, 800],
        full=[],
        body=[dict(text=brand, confidence=0.99, box=[500, 300, 580, 340])],
        pairs=[
            dict(
                full_text=token,
                body_text=token,
                full_box=[420, 320, 460, 350],
                body_box=[420, 320, 460, 350],
            )
        ],
    )


def test_branded_conflict_preserves_reading_and_reviewable_family_hint():
    evidence = branded_evidence("T5050", "nh_t5000", "NEW HOLLAND")
    result = predict_hierarchy(bundle(), np.zeros(2), evidence)
    assert result["family_candidate"] == "nh_t5000_family"
    assert result["classifier_family_candidate"] == "mtz_80_family"
    assert result["visible_model_marking"] == "T5050"
    assert result["family"] is None and result["exact_model"] is None
    assert result["marking_conflict"] and result["candidate_source"] == "branded_marking"


def test_unbranded_digits_cannot_override_visual_family():
    evidence = branded_evidence("T5050", "nh_t5000", "")
    result = predict_hierarchy(bundle(), np.zeros(2), evidence)
    assert result["family_candidate"] == "mtz_80_family"
    assert result["visible_model_marking"] == "T5050"
    assert result["family"] is None


def test_distant_brand_or_low_confidence_cannot_supply_family_hint():
    from src.models.hierarchy import branded_marking_family

    evidence = branded_evidence()
    evidence["body"][0]["box"] = [900, 20, 990, 50]
    assert branded_marking_family(evidence) is None
    evidence = branded_evidence()
    evidence["body"][0]["confidence"] = 0.89
    assert branded_marking_family(evidence) is None


def test_model_token_must_agree_with_suggested_family():
    from src.models.hierarchy import branded_marking_family

    assert branded_marking_family(branded_evidence("T5050", "mtz_82", "MTZ")) is None


def test_corner_number_is_not_hood_marking():
    from src.models.hierarchy import branded_marking_family

    evidence = branded_evidence()
    evidence["pairs"][0]["full_box"] = [420, 740, 460, 780]
    evidence["body"][0]["box"] = [500, 740, 580, 780]
    assert branded_marking_family(evidence) is None


def test_corroborated_unknown_nh_series_withholds_known_family():
    marking = dict(status="unconfirmed", image_size=[1000, 800])
    lines = [
        dict(text="M135", confidence=0.999, box=[400, 300, 450, 330]),
        dict(text="NEW HOLLAND", confidence=0.99, box=[470, 300, 600, 330]),
    ]
    marking["full"] = lines
    marking["body"] = lines
    result = predict_hierarchy(bundle("nh_t5000_family"), np.zeros(2), marking)
    assert result["family_candidate"] == "unknown" and result["family"] is None
    assert result["visible_model_marking"] == "M135"
    assert result["candidate_source"] == "unsupported_marking"
    assert result["exact_model"] is None


def test_unsupported_nh_model_requires_brand_in_both_views():
    from src.models.hierarchy import unsupported_nh_marking

    model = dict(text="M135", confidence=0.999, box=[400, 300, 450, 330])
    brand = dict(text="NEW HOLLAND", confidence=0.99, box=[470, 300, 600, 330])
    assert (
        unsupported_nh_marking(
            dict(status="unconfirmed", image_size=[1000, 800], full=[model, brand], body=[model])
        )
        is None
    )


def test_supported_t5000_is_not_an_unsupported_series():
    from src.models.hierarchy import unsupported_nh_marking

    lines = [
        dict(text="T5050", confidence=0.999, box=[400, 300, 450, 330]),
        dict(text="NEW HOLLAND", confidence=0.99, box=[470, 300, 600, 330]),
    ]
    assert (
        unsupported_nh_marking(
            dict(status="unconfirmed", image_size=[1000, 800], full=lines, body=lines)
        )
        is None
    )
