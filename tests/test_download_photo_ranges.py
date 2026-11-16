"""HTTP range integrity checks for dataset downloads."""

from types import SimpleNamespace

import pytest

from scripts.download_photo_ranges import range_total


def test_incomplete_or_wrong_offset_range_is_not_appended():
    correct = SimpleNamespace(
        status_code=206, headers={"Content-Range": "bytes 16-19/20"}, content=b"data"
    )
    assert range_total(correct, 16) == 20
    with pytest.raises(ValueError, match="range response"):
        range_total(correct, 0)
    correct.content = b"bad"
    with pytest.raises(ValueError, match="range response"):
        range_total(correct, 16)


def test_ignored_range_is_not_appended_as_a_chunk():
    response = SimpleNamespace(status_code=200, headers={}, content=b"full image")
    with pytest.raises(ValueError, match="honor"):
        range_total(response, 16)
