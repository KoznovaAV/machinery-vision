"""Protect extended-class taxonomy and keep the pilot explicitly non-automatic."""

import pytest

from scripts.predict_field_pilot import review_output
from scripts.train_field_model_pilot import assert_independent, relabel_old_negative


def row(key="a", group="one", sha="one", model="other"):
    return dict(key=key, group=group, sha256=sha, model=model)


def test_relabel_only_previously_other_known_categories():
    old = row(key="negative-train/photo")
    assert relabel_old_negative(old, {"photo": "DT-75"})["model"] == "dt_75"
    assert old["model"] == "other"
    assert relabel_old_negative(old, {"photo": "new holland"})["model"] == "other"
    assert (
        relabel_old_negative(row(key="photo", model="mtz_80"), {"photo": "DT-75"})["model"]
        == "mtz_80"
    )


def test_pixel_leakage_rejected():
    with pytest.raises(ValueError, match="pixel leakage"):
        assert_independent([row()], [row("b", "two", "one")])


def test_group_leakage_rejected():
    with pytest.raises(ValueError, match="group leakage"):
        assert_independent([row()], [row("b", "one", "two")])


def test_new_class_never_becomes_automatic_answer():
    result = review_output([1, 5], ["other", "acros_550"])
    assert result["visual_candidate"] == "acros_550"
    assert result["status"] == "manual_review" and result["model"] is None


def test_invalid_class_dimensions_rejected():
    with pytest.raises(ValueError):
        review_output([1, 2], ["acros_550"])


def test_repeated_frame_with_different_key_rejected():
    with pytest.raises(ValueError, match="Repeated pixel"):
        assert_independent([row(), row("b", "two", "one")], [])
