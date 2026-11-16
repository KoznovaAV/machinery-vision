"""Tests of conservative marking parsing and region geometry."""

import pytest
from PIL import Image

from src.models.parts import lower_proposal, marking_labels


@pytest.mark.parametrize(
    "text,expected",
    [
        ("BELARUS 820.1", ["mtz_820"]),
        ("82.1", ["mtz_82"]),
        ("80.1", ["mtz_80"]),
        ("T5050", ["nh_t5000"]),
        ("T 5070", ["nh_t5000"]),
        ("T5080", []),
        ("12820", []),
        ("X820A", []),
        ("820 82", ["mtz_82", "mtz_820"]),
        ("TD65D", ["outside_supported"]),
        ("CLAAS ARION 540", ["outside_supported"]),
    ],
)
def test_marking_boundaries(text, expected):
    assert marking_labels(text) == expected


def test_lower_proposal_keeps_both_ends():
    image = Image.new("RGB", (200, 100))
    assert lower_proposal(image).size == (200, 52)


def test_front_region_geometry():
    from src.models.parts import front_region

    image = Image.new("RGB", (100, 100))
    assert front_region(image, "L").size == (62, 55)
    assert front_region(image, "R").size == (62, 55)
    with pytest.raises(ValueError):
        front_region(image, "unknown")


def test_parts_evidence_can_only_veto():
    from src.models.parts import merge_evidence

    base = dict(status="experimental_match", model="nh_t5000", suggestion="nh_t5000")
    assert merge_evidence(base, "nh_t5000", "nh_t5000", [])["result"]["model"] == "nh_t5000"
    assert merge_evidence(base, "mtz_82", "nh_t5000", [])["result"]["model"] is None
    assert (
        merge_evidence(base, "nh_t5000", "nh_t5000", ["outside_supported"])["result"]["model"]
        is None
    )
    family = dict(status="family_only", model=None, suggestion="mtz_80")
    assert merge_evidence(family, "mtz_80", "mtz_80", ["mtz_80"])["result"]["model"] is None


def test_conflicting_marking_removes_family_claim():
    from src.models.parts import merge_evidence

    base = dict(status="family_only", model=None, suggestion="mtz_80", family="mtz_belarus")
    result = merge_evidence(base, "mtz_80", "mtz_80", ["nh_t5000"])["result"]
    assert result["status"] == "manual_review"
    assert result["family"] is None


def test_equipment_number_does_not_override_explicit_model():
    from src.models.parts import marking_candidates

    lines = [
        dict(text="T5070", confidence=0.99, candidates=["nh_t5000"]),
        dict(text="82:", confidence=0.99, candidates=["mtz_82"]),
    ]
    assert marking_candidates(lines) == ["nh_t5000"]
    assert marking_candidates(lines[1:]) == []
    assert marking_candidates([dict(text="820", confidence=0.99, candidates=["mtz_820"])]) == [
        "mtz_820"
    ]
