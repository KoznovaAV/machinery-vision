"""Regressions for implements and independent visible condition attributes."""

import numpy as np

from src.models import mvp_inference as mvp
from tests.test_machinery_hierarchy import bundle, constant


def test_high_confidence_implement_is_rejected(monkeypatch):
    head = constant(["attachment", "machine"], "attachment")
    monkeypatch.setattr(
        mvp, "load_optional_bundle", lambda name: dict(head=head, reject_threshold=0.85)
    )
    accepted, evidence = mvp.verify_machine(np.zeros(2))
    assert not accepted and evidence["label"] == "attachment"


def test_separate_machine_is_kept(monkeypatch):
    head = constant(["attachment", "machine"], "machine")
    monkeypatch.setattr(
        mvp, "load_optional_bundle", lambda name: dict(head=head, reject_threshold=0.85)
    )
    assert mvp.verify_machine(np.zeros(2))[0]


def test_dirt_and_snow_can_coexist_without_known_family(monkeypatch):
    heads = {attribute: constant(["0", "1"], "1") for attribute in ("dirty", "snow")}
    calibration = {attribute: {"1": {"threshold": 0.8}} for attribute in heads}
    monkeypatch.setattr(
        mvp, "load_optional_bundle", lambda name: dict(heads=heads, calibration=calibration)
    )
    result = mvp.assess_conditions(np.zeros(2), np.zeros(2))
    assert result["dirty"]["value"] is True and result["snow"]["value"] is True
    assert result["rain"]["status"] == "not_assessed"


def test_missing_condition_head_does_not_imply_clean(monkeypatch):
    monkeypatch.setattr(mvp, "load_optional_bundle", lambda name: None)
    result = mvp.assess_conditions(np.zeros(2), np.zeros(2))
    assert all(value["status"] == "not_assessed" for value in result.values())


def test_uncalibrated_condition_is_withheld(monkeypatch):
    head = constant(["0", "1"], "1")
    monkeypatch.setattr(
        mvp,
        "load_optional_bundle",
        lambda name: dict(heads={"dirty": head}, calibration={"dirty": {"1": {"threshold": None}}}),
    )
    result = mvp.assess_conditions(np.zeros(2), np.zeros(2))
    assert result["dirty"]["value"] is None


def test_type_agreement_can_resolve_detector_category_noise():
    assert mvp.resolve_detected_type(bundle(), np.zeros(2), "combine_harvester") == "tractor"
    assert mvp.resolve_detected_type(bundle(), np.zeros(2), "bulldozer") == "bulldozer"


def test_uncertain_crop_is_not_promoted_to_second_machine(monkeypatch):
    head = dict(
        classes=["attachment", "machine"], coef=[[0.0, 0.0], [0.0, 0.0]], intercept=[0.0, 0.1]
    )
    monkeypatch.setattr(
        mvp, "load_optional_bundle", lambda name: dict(head=head, reject_threshold=0.85)
    )
    assert not mvp.verify_machine(np.zeros(2))[0]


def test_ambiguous_detector_text_does_not_select_bulldozer_by_word_order():
    from src.models.machinery_regions import select_regions

    raw = [dict(box=[10, 10, 90, 90], score=0.8, label="tractor bulldozer")]
    regions = select_regions(raw, 100, 100)
    assert len(regions) == 1 and regions[0]["type_candidate"] is None
    assert mvp.resolve_detected_type(bundle(), np.zeros(2), None) == "tractor"


def test_explicit_bundle_is_used_for_offline_comparison(monkeypatch):
    monkeypatch.setattr(
        mvp,
        "load_optional_bundle",
        lambda name: (_ for _ in ()).throw(AssertionError("unexpected active model")),
    )
    head = constant(["0", "1"], "1")
    candidate = dict(heads={"dirty": head}, calibration={"dirty": {"1": {"threshold": 0.8}}})
    result = mvp.assess_conditions(np.zeros(2), np.zeros(2), bundle=candidate)
    assert result["dirty"]["value"] is True
    assert result["rain"]["status"] == "not_assessed"


def test_high_confidence_absence_of_dirt_can_still_require_review():
    head = constant(["0", "1"], "0")
    candidate = dict(heads={"dirty": head}, calibration={"dirty": {"0": {"threshold": None}}})
    result = mvp.assess_conditions(np.zeros(2), np.zeros(2), bundle=candidate)
    assert result["dirty"]["status"] == "not_assessed"
    assert result["dirty"]["value"] is None


def test_partial_specialist_recovers_only_edge_machine(monkeypatch):
    classifier = dict(
        head=constant(["attachment", "machine"], "attachment"),
        partial_head=constant(["attachment", "machine"], "machine"),
        partial_calibration={"machine": {"threshold": 0.95}},
    )
    monkeypatch.setattr(mvp, "load_optional_bundle", lambda name: classifier)
    accepted, evidence = mvp.verify_machine(np.zeros(2), box=[0, 20, 80, 90], image_size=(200, 200))
    assert accepted and evidence["partial_rescue"]
    assert not mvp.verify_machine(np.zeros(2), box=[20, 20, 80, 90], image_size=(200, 200))[0]


def test_fragment_uncertainty_cannot_restore_attachment(monkeypatch):
    classifier = dict(
        head=constant(["attachment", "machine"], "attachment"),
        partial_head=dict(classes=["attachment", "machine"], coef=[[0, 0]], intercept=[0.5]),
        partial_calibration={"machine": {"threshold": 0.95}},
    )
    monkeypatch.setattr(mvp, "load_optional_bundle", lambda name: classifier)
    accepted, _ = mvp.verify_machine(np.zeros(2), box=[0, 20, 80, 90], image_size=(200, 200))
    assert not accepted


def test_trailer_reference_requires_calibration():
    assert not mvp.verify_trailer(np.zeros(2), bundle={})[0]
    assert not mvp.verify_trailer(np.zeros(2), bundle={"trailer_reference": {"threshold": None}})[0]


def test_trailer_reference_prefers_attachment_only_at_calibrated_margin(monkeypatch):
    references = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    monkeypatch.setattr(
        mvp,
        "_trailer_references",
        lambda path, sha: (
            references,
            np.array(["attachment", "machine"]),
            np.array(["trailer-example", "tractor-example"]),
        ),
    )
    b = {"trailer_reference": {"path": "unused", "sha256": "unused", "threshold": 0.1}}
    veto, evidence = mvp.verify_trailer(np.array([1.0, 0.0]), bundle=b)
    assert veto and evidence["score_kind"] == "cosine_margin"
    assert evidence["attachment_reference"] == "trailer-example"
    assert not mvp.verify_trailer(np.array([0.0, 1.0]), bundle=b)[0]
    assert not mvp.verify_trailer(np.array([1.0, 1.0]), bundle=b)[0]


def test_invalid_or_missing_trailer_references_cannot_remove_machine(monkeypatch):
    b = {"trailer_reference": {"path": "unused", "sha256": "unused", "threshold": 0.1}}

    def missing(*args):
        raise OSError("Missing weights")

    monkeypatch.setattr(mvp, "_trailer_references", missing)
    assert not mvp.verify_trailer(np.ones(2), bundle=b)[0]


def test_changed_trailer_reference_file_is_rejected(tmp_path, monkeypatch):
    target = tmp_path / "vectors.npz"
    np.savez_compressed(
        target,
        vectors=np.eye(2),
        labels=np.array(["attachment", "machine"]),
        keys=np.array(["a", "b"]),
    )
    import pytest

    monkeypatch.setattr(mvp, "ROOT", tmp_path)
    with pytest.raises(ValueError):
        mvp._trailer_references(str(target), "wrong-sha")


def test_linear_attachment_confirmation_requires_reference_agreement(monkeypatch):
    refs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    monkeypatch.setattr(
        mvp,
        "_trailer_references",
        lambda p, s: (refs, np.array(["attachment", "machine"]), np.array(["a", "m"])),
    )
    cfg = {
        "path": "unused",
        "sha256": "unused",
        "threshold": 0.9,
        "minimum_margin": 0.03,
        "supplement_head": constant(["attachment", "machine"], "attachment"),
        "supplement_calibration": {
            "attachment": {"threshold": 0.95},
            "machine": {"threshold": 0.95},
        },
        "supplement_minimum_margin": 0.03,
    }
    b = {"trailer_reference": cfg}
    assert mvp.verify_trailer(np.array([0.8, 0.6]), bundle=b)[0]
    rejected, evidence = mvp.verify_trailer(np.array([0.69, 0.71]), bundle=b)
    assert not rejected and evidence["uncertain_role"]
    assert not mvp.verify_trailer(np.array([0.0, 1.0]), bundle=b)[0]


def test_balanced_reference_query_uses_both_encoder_blocks(monkeypatch):
    a = np.zeros(1152, dtype=np.float32)
    a[0] = 1
    a[384] = 1
    a /= np.linalg.norm(a)
    m = np.zeros(1152, dtype=np.float32)
    m[1] = 1
    m[385] = 1
    m /= np.linalg.norm(m)
    monkeypatch.setattr(
        mvp,
        "_trailer_references",
        lambda p, s: (np.stack([a, m]), np.array(["attachment", "machine"]), np.array(["a", "m"])),
    )
    value = np.zeros(1152)
    value[0] = 100
    value[384] = 1
    b = {
        "trailer_reference": {
            "path": "unused",
            "sha256": "unused",
            "threshold": 0.5,
            "normalization": "per_encoder_l2",
        }
    }
    ok, evidence = mvp.verify_trailer(value, bundle=b)
    assert ok and evidence["margin"] > 0.99
