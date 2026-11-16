"""Checks of manually supervised detection split and annotation geometry."""

import pytest

from scripts.train_tractor_parts_detector import validate_annotations


def row(key="a", group="g", index=0, box=None, role="train"):
    return dict(
        key=key, group=group, index=index, role=role, boxes=[] if box is None else [dict(xyxy=box)]
    )


def test_control_never_enters_detector_training():
    assert validate_annotations([row(), row("b", "h", 1, role="test")]) == [row()]


def test_detector_group_leakage_rejected():
    with pytest.raises(ValueError, match="leakage"):
        validate_annotations([row(), row("b", "g", 1)])


@pytest.mark.parametrize(
    "box", [[-0.1, 0, 0.5, 1], [0, 0, 1.1, 1], [0.5, 0, 0.4, 1], [0, 0.8, 1, 0.8]]
)
def test_invalid_manual_box_rejected(box):
    with pytest.raises(ValueError, match="annotation box"):
        validate_annotations([row(box=box)])


def test_repeated_training_key_rejected():
    with pytest.raises(ValueError, match="Repeated"):
        validate_annotations([row(), row("a", "h", 3)])
