import pytest
from src.models.trailer_filter import split_trailers


def regions():
    return [dict(box=[0, 0, 200, 200]), dict(box=[220, 0, 320, 150])]


def semantic(label, box, score):
    return dict(label=label, box=box, score=score)


def test_trailer_removed_with_aligned_separate_tractor():
    r = regions()
    keep, drop = split_trailers(
        r, [semantic("tractor", r[0]["box"], 0.7), semantic("trailer", r[1]["box"], 0.6)]
    )
    assert keep == r[:1] and drop[0]["reason"] == "semantic_trailer"


def test_two_real_tractors_not_suppressed_by_overlapping_trailer():
    r = regions()
    keep, drop = split_trailers(
        r,
        [
            semantic("tractor", r[0]["box"], 0.7),
            semantic("tractor", r[1]["box"], 0.65),
            semantic("trailer", r[1]["box"], 0.6),
        ],
    )
    assert keep == r and not drop


def test_partial_tractor_retained_when_trailer_box_is_spatially_different():
    r = regions()
    keep, drop = split_trailers(
        r, [semantic("tractor", r[0]["box"], 0.7), semantic("trailer", [200, 0, 450, 200], 0.9)]
    )
    assert keep == r and not drop


def test_trailer_inside_whole_tractor_box_does_not_remove_whole_machine():
    r = regions()
    r[0]["box"] = [0, 0, 320, 200]
    keep, drop = split_trailers(
        r, [semantic("tractor", r[0]["box"], 0.7), semantic("trailer", r[1]["box"], 0.6)]
    )
    assert keep == r[:1] and len(drop) == 1


@pytest.mark.parametrize("label,score", [("trailer", 0.2), ("tractor trailer", 0.8)])
def test_weak_or_ambiguous_semantics_retained(label, score):
    r = regions()
    assert split_trailers(
        r, [semantic("tractor", r[0]["box"], 0.7), semantic(label, r[1]["box"], score)]
    ) == (r, [])


def test_trailer_label_without_other_supported_machine_does_not_delete_everything():
    r = regions()
    assert split_trailers(r, [semantic("trailer", r[1]["box"], 0.6)]) == (r, [])


def test_single_object_unchanged():
    r = regions()[:1]
    assert split_trailers(r, [semantic("trailer", r[0]["box"], 0.9)]) == (r, [])


def test_standalone_semantic_proposal_requires_explicit_image_verification_mode():
    r = regions()[:1]
    s = [semantic("trailer", r[0]["box"], 0.8)]
    assert split_trailers(r, s) == (r, [])
    keep, drop = split_trailers(r, s, require_other_powered=False)
    assert not keep and drop[0]["reason"] == "semantic_trailer"


def test_composite_box_of_tractor_and_confirmed_tank_is_removed():
    from src.models.trailer_filter import remove_composite_views

    kept = [(0, dict(box=[0, 0, 60, 100], score=0.6)), (1, dict(box=[0, 0, 100, 100], score=0.4))]
    attachments = [dict(bbox=[60, 0, 100, 100], reason="semantic_trailer")]
    result, dropped = remove_composite_views(kept, attachments)
    assert result == kept[:1] and dropped[0]["reason"] == "composite_duplicate"


def test_composite_filter_preserves_stronger_foreground_machine():
    from src.models.trailer_filter import remove_composite_views

    kept = [(0, dict(box=[0, 0, 60, 100], score=0.4)), (1, dict(box=[0, 0, 100, 100], score=0.6))]
    assert remove_composite_views(
        kept, [dict(bbox=[60, 0, 100, 100], reason="semantic_trailer")]
    ) == (kept, [])


def test_uncertain_attachment_does_not_authorize_composite_removal():
    from src.models.trailer_filter import remove_composite_views

    kept = [(0, dict(box=[0, 0, 60, 100], score=0.6)), (1, dict(box=[0, 0, 100, 100], score=0.4))]
    assert remove_composite_views(
        kept, [dict(bbox=[60, 0, 100, 100], reason="ambiguous_powered_or_attachment")]
    ) == (kept, [])
