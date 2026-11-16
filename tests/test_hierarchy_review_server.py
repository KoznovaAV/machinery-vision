from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from scripts.web import create_app


def photo():
    stream = BytesIO()
    Image.new("RGB", (80, 80), "green").save(stream, format="JPEG")
    return stream.getvalue()


def test_upload_interface_has_explicit_review_warning(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "Экспериментальный режим" in response.text


def test_invalid_photo_is_rejected_before_inference(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert (
            client.post(
                "/predict", files={"file": ("bad.jpg", b"not a photo", "image/jpeg")}
            ).status_code
            == 400
        )


def test_uploaded_photo_reaches_separate_predictor_and_local_artifact(tmp_path):
    seen = []

    def predictor(source, bundle, output):
        seen.append(source)
        return dict(objects=[])

    with TestClient(create_app(tmp_path, predictor=predictor)) as client:
        response = client.post(
            "/predict", files={"file": ("../../outside.jpg", photo(), "image/jpeg")}
        )
    assert response.status_code == 200 and len(seen) == 1
    assert seen[0].is_relative_to(tmp_path)
    assert list(tmp_path.glob("*/review.html"))


def test_oversized_photo_is_rejected(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert (
            client.post(
                "/predict", files={"file": ("large.jpg", b"x" * 8_000_001, "image/jpeg")}
            ).status_code
            == 413
        )


def test_inference_failure_does_not_leak_internal_error_to_web_page(tmp_path):
    def failing(source, bundle, output):
        raise RuntimeError("private internal diagnostic")

    with TestClient(create_app(tmp_path, predictor=failing)) as client:
        response = client.post("/predict", files={"file": ("photo.jpg", photo(), "image/jpeg")})
    assert response.status_code == 500
    assert "private internal diagnostic" not in response.text


def test_standalone_trailer_is_explained_without_machine_card():
    from scripts.render_review import render

    page = render({"objects": [], "ignored_regions": [{"reason": "semantic_trailer"}]})
    assert "Обнаружен прицеп" in page
    assert "Машина 1" not in page


def test_uncertain_region_is_visible_without_machine_identity():
    from scripts.render_review import render

    page = render({"objects": [], "uncertain_regions": [{"bbox": [0, 0, 10, 10]}]})
    assert "Неподтверждённая область" in page and "Машина 1" not in page
