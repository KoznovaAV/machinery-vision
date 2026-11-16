"""Bounded download checks without external network access."""

from io import BytesIO

import pytest

from scripts.download_stream_ranges import BLOCKED, download, read_part


class Response:
    def __init__(self, range_value="bytes 0-3/4", body=b"abcd", status=206):
        self.status_code = status
        self.headers = {"Content-Range": range_value}
        self.raw = BytesIO(body)
        self.closed = False

    def close(self):
        self.closed = True

    def raise_for_status(self):
        pass


class Session:
    def __init__(self, response):
        self.response = response

    def get(self, *args, **kwargs):
        return self.response


def test_stream_reads_exact_declared_range_and_closes_response():
    response = Response()
    payload, total, _, _ = read_part(Session(response), "https://example.org/photo", 0)
    assert payload == b"abcd" and total == 4 and response.closed


def test_wrong_range_cannot_be_appended():
    response = Response("bytes 0-3/8")
    with pytest.raises(ValueError):
        read_part(Session(response), "https://example.org/photo", 4)
    assert response.closed


def test_incomplete_response_is_not_a_photo():
    with pytest.raises(ValueError):
        read_part(Session(Response(body=b"ab")), "https://example.org/photo", 0)


def test_denied_host_is_deferred_not_bypassed():
    with pytest.raises(ValueError):
        read_part(Session(Response(status=403)), "https://denied.invalid/photo", 0)
    assert "denied.invalid" in BLOCKED
    BLOCKED.discard("denied.invalid")


def test_download_path_cannot_escape_collection(tmp_path):
    with pytest.raises(ValueError):
        download(dict(file="../escape.jpg", id="x", url="https://example.org/photo"), tmp_path)


def test_default_size_bound_rejects_oversized_photo():
    response = Response('bytes 0-3/2500000')
    with pytest.raises(ValueError, match='Photo too large'):
        read_part(Session(response), 'https://example.org/photo', 0)
    assert response.closed


def test_explicit_larger_bound_accepts_valid_bounded_range():
    response = Response('bytes 0-3/2500000')
    payload, total, _, _ = read_part(Session(response), 'https://example.org/photo', 0, max_bytes=3000000)
    assert payload == b'abcd' and total == 2500000 and response.closed
