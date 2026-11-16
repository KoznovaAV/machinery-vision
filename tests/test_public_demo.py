"""Public demo limits and request-data isolation."""

import asyncio
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image, ImageOps

from scripts.web import create_app
from src.web.public import PublicRequestLimits, RETENTION_SECONDS, purge_expired


def payload(size=(80, 90), *, metadata=False):
    buffer = BytesIO()
    image = Image.new("RGB", size, (40, 100, 70))
    exif = Image.Exif()
    if metadata:
        exif[0x010E] = "private image description"
        exif[0x0112] = 6
    image.save(buffer, format="JPEG", exif=exif)
    return buffer.getvalue()


def test_public_input_has_identical_pixels_and_no_exif(tmp_path):
    data = payload(metadata=True)
    seen = []

    def predictor(source, bundle, output):
        with Image.open(source) as image:
            assert not image.getexif()
            seen.append(np.asarray(image).copy())
        return {"objects": []}

    with Image.open(BytesIO(data)) as raw:
        expected = np.asarray(ImageOps.exif_transpose(raw).convert("RGB"))
    with TestClient(create_app(tmp_path, predictor=predictor, public_demo=True)) as client:
        response = client.post("/predict", files={"file": ("photo.jpg", data, "image/jpeg")})
    assert response.status_code == 200
    assert np.array_equal(seen[0], expected)
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-robots-tag"] == "noindex, nofollow"


def test_public_files_are_images_only_and_json_is_private(tmp_path):
    folder = tmp_path / ("a" * 32)
    folder.mkdir()
    (folder / "prediction.json").write_text('{"secret":"internal"}')
    Image.new("RGB", (80, 80)).save(folder / "machine.png")
    with TestClient(create_app(tmp_path, public_demo=True)) as client:
        assert client.get("/artifacts/" + folder.name + "/prediction.json").status_code == 404
        assert client.get("/artifacts/" + folder.name + "/machine.png").status_code == 200
        assert client.get("/docs").status_code == 404


def test_expired_image_is_not_served_even_before_cleanup(tmp_path):
    folder = tmp_path / ("b" * 32)
    folder.mkdir()
    Image.new("RGB", (80, 80)).save(folder / "machine.png")
    stamp = time.time() - RETENTION_SECONDS - 10
    os.utime(folder, (stamp, stamp))
    client = TestClient(create_app(tmp_path, public_demo=True))
    assert client.get("/artifacts/" + folder.name + "/machine.png").status_code == 404
    assert (folder / "machine.png").exists()
    client.close()


def test_cleanup_only_removes_expired_request_folders(tmp_path):
    names = ["a" * 32, "b" * 32, "not-a-request"]
    folders = [tmp_path / name for name in names]
    for folder in folders:
        folder.mkdir()
        (folder / "file.txt").write_text("data")
        os.utime(folder, (0, 0))
    purge_expired(tmp_path, active={folders[1]}, now=RETENTION_SECONDS + 1)
    assert not folders[0].exists()
    assert folders[1].exists() and folders[2].exists()


def test_public_rejects_oversized_body_before_parsing(tmp_path):
    with TestClient(create_app(tmp_path, public_demo=True)) as client:
        response = client.post("/predict", content=b"x" * 8_100_001)
    assert response.status_code == 413
    assert not list(tmp_path.iterdir())


def test_chunked_body_has_the_same_limit_and_releases_slot():
    called = []

    async def inner(scope, receive, send):
        called.append(True)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"OK"})

    middleware = PublicRequestLimits(inner, max_body_bytes=10)

    async def request(chunks):
        messages = iter(
            [
                {"type": "http.request", "body": chunk, "more_body": i < len(chunks) - 1}
                for i, chunk in enumerate(chunks)
            ]
        )
        responses = []

        async def receive():
            return next(messages)

        async def send(message):
            responses.append(message)

        await middleware(
            {"type": "http", "method": "POST", "path": "/predict", "headers": []}, receive, send
        )
        return next(x["status"] for x in responses if x["type"] == "http.response.start")

    assert asyncio.run(request([b"12345678", b"abcdefgh"])) == 413
    assert not called
    assert asyncio.run(request([b"123"])) == 200


def test_busy_public_demo_rejects_instead_of_queuing(tmp_path):
    entered, release = threading.Event(), threading.Event()

    def predictor(source, bundle, output):
        entered.set()
        assert release.wait(5)
        return {"objects": []}

    data = payload()
    with TestClient(create_app(tmp_path, predictor=predictor, public_demo=True)) as client:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                client.post, "/predict", files={"file": ("a.jpg", data, "image/jpeg")}
            )
            try:
                assert entered.wait(3)
                response = client.post("/predict", files={"file": ("b.jpg", data, "image/jpeg")})
                assert response.status_code == 429
            finally:
                release.set()
            assert future.result().status_code == 200


def test_public_image_dimensions_are_bounded(tmp_path):
    with TestClient(create_app(tmp_path, public_demo=True)) as client:
        response = client.post(
            "/predict", files={"file": ("wide.jpg", payload((8193, 64)), "image/jpeg")}
        )
    assert response.status_code == 413
    assert not list(tmp_path.iterdir())
