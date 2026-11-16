"""Protect groups, image identity and cached labels during dataset updates."""

import hashlib, json
import numpy as np
import pytest
from src.data.manifest import CONTROLS, assert_independent, audit


def test_another_view_of_the_same_source_cannot_enter_control():
    with pytest.raises(ValueError, match="group"):
        assert_independent([{"group": "one", "sha256": "a"}], [{"group": "one", "sha256": "b"}])


def test_renamed_copy_cannot_enter_control():
    with pytest.raises(ValueError, match="image"):
        assert_independent(
            [{"group": "one", "sha256": "same"}], [{"group": "two", "sha256": "same"}]
        )


@pytest.fixture
def dataset(tmp_path):
    def save(name, value):
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value), encoding="utf8")

    image = tmp_path / "datasets/images/a.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"photo")
    row = {
        "key": "one",
        "path": "datasets/images/a.jpg",
        "sha256": hashlib.sha256(b"photo").hexdigest(),
        "group": "one",
        "model": "mtz_80",
    }
    data = {k: [] for k in CONTROLS}
    data.update(train=[row], family_only_train=[], condition_rows=[])
    save("datasets/manifest.json", data)
    save("datasets/regions.json", {"train": []})
    for name in ("region_control", "annotations", "scenes"):
        save("datasets/" + name + ".json", [])
    save("datasets/features/index.json", {"one": row["sha256"]})
    np.savez_compressed(
        tmp_path / "datasets/features/recognition.npz",
        keys=np.array(["full/one", "crop/one"]),
        values=np.zeros((2, 1152)),
    )
    np.savez_compressed(
        tmp_path / "datasets/features/regions.npz",
        keys=np.array([], dtype="str"),
        labels=np.array([], dtype="str"),
        groups=np.array([], dtype="str"),
        vectors=np.zeros((0, 1152)),
    )
    return tmp_path, save, data


def test_current_dataset_is_accepted(dataset):
    root, _, _ = dataset
    assert audit(root)["passed"]


def test_changed_image_requires_new_features(dataset):
    root, _, _ = dataset
    (root / "datasets/images/a.jpg").write_bytes(b"other")
    assert not audit(root)["passed"]


def test_image_path_cannot_escape_project(dataset):
    root, save, data = dataset
    data["train"][0]["path"] = "../outside.jpg"
    save("datasets/manifest.json", data)
    assert not audit(root)["passed"]


def test_feature_index_must_follow_image_identity(dataset):
    root, save, _ = dataset
    save("datasets/features/index.json", {"one": "different"})
    assert any("Stale features" in e for e in audit(root)["errors"])
