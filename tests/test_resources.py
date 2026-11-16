"""Resource integrity and model compatibility contracts."""

import hashlib
import json
import pytest
from scripts.verify_resources import verify


@pytest.fixture
def bundle(tmp_path):
    (tmp_path / "image.jpg").write_bytes(b"image")
    (tmp_path / "head.json").write_bytes(b"model")

    def record(name):
        data = (tmp_path / name).read_bytes()
        return {
            "path": name,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    manifest = {
        "format": "machinery-vision-resources-v1",
        "files": [record("image.jpg")],
        "compatible_artifacts": [record("head.json")],
    }
    (tmp_path / "RESOURCES_MANIFEST.json").write_text(json.dumps(manifest))
    return tmp_path, manifest


def test_complete_matching_resources_pass(bundle):
    root, _ = bundle
    assert verify(root)["passed"]


def test_changed_image_fails_verification(bundle):
    root, _ = bundle
    (root / "image.jpg").write_bytes(b"wrong")
    assert not verify(root)["passed"]


def test_different_model_cannot_use_matching_resources(bundle):
    root, _ = bundle
    (root / "head.json").write_bytes(b"other")
    assert not verify(root)["passed"]


def test_manifest_cannot_read_files_outside_project(bundle):
    root, manifest = bundle
    manifest["files"][0]["path"] = "../image.jpg"
    (root / "RESOURCES_MANIFEST.json").write_text(json.dumps(manifest))
    assert not verify(root)["passed"]
