"""Dataset split stability protects external control from later training reuse."""

import pytest

from scripts.audit_round4_candidates import canonical_source
from scripts.freeze_future_field_models import preserve_group_splits


def candidate(i, g, sha="a", model="mtz_1523"):
    return dict(id=i, group=g, sha256=sha, model=model)


def test_appending_sources_preserves_previous_control():
    previous = [dict(candidate("one", "g1"), split="future_test")]
    new = [
        candidate("one", "g1"),
        candidate("another_view", "g1", "b"),
        candidate("new_source", "g2", "c"),
    ]
    result = preserve_group_splits(new, previous)
    assert result[0]["split"] == result[1]["split"] == "future_test"


def test_changed_registered_label_rejected():
    previous = [dict(candidate("one", "g1"), split="future_test")]
    with pytest.raises(ValueError, match="changed"):
        preserve_group_splits([candidate("one", "g1", model="mtz_1221")], previous)


def test_source_normalization_retains_identity_query():
    assert (
        canonical_source("http://www.example.com/ad/?id=7&utm_source=x#photo")
        == "https://example.com/ad?id=7"
    )
    assert canonical_source("https://example.com/ad?id=7") != canonical_source(
        "https://example.com/ad?id=8"
    )


def test_registered_control_cannot_disappear_silently():
    previous = [dict(candidate("one", "g1"), split="future_test")]
    with pytest.raises(ValueError, match="missing"):
        preserve_group_splits([], previous)
