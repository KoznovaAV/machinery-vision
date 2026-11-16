"""Load one explicit dataset; source groups and image SHA define split identity."""

import hashlib
import json
from pathlib import Path
import numpy as np
from src.config.runtime import ROOT

CONTROLS = ("test", "winter_test", "fresh_test", "new_photos_test", "additional_test")


def read(path):
    return json.loads(Path(path).read_bytes())


def load_dataset(root=ROOT):
    return read(root / "datasets/manifest.json")


def load_features(root=ROOT):
    z = np.load(root / "datasets/features/recognition.npz", allow_pickle=False)
    full, crop = {}, {}
    for key, value in zip(z["keys"], z["values"], strict=True):
        key = str(key)
        (full if key.startswith("full/") else crop)[key[5:]] = value
    return full, crop


def assert_independent(train, control):
    if {r["group"] for r in train} & {r["group"] for r in control}:
        raise ValueError("A source/machine group appears in training and control")
    left = {r["sha256"] for r in train if r.get("sha256")}
    right = {r["sha256"] for r in control if r.get("sha256")}
    if left & right:
        raise ValueError("An image appears in training and control")


def audit(root=ROOT, verify_hashes=True):
    root = Path(root).resolve()
    data = load_dataset(root)
    training = data["train"] + data["family_only_train"]
    controls = [r for name in CONTROLS for r in data[name]]
    assert_independent(training, controls)
    assert_independent(
        [r for r in data["condition_rows"] if r["role"] == "train"],
        [r for r in data["condition_rows"] if r["role"] == "test"],
    )
    region_rows = read(root / "datasets/regions.json")["train"]
    assert_independent(region_rows, controls)
    errors, seen = [], {}
    bank = np.load(root / "datasets/features/regions.npz", allow_pickle=False)
    bank_rows = {
        str(k): (str(label), str(group))
        for k, label, group in zip(bank["keys"], bank["labels"], bank["groups"], strict=True)
    }
    manifest_rows = {r["key"]: (r["model"], r["group"]) for r in region_rows}
    if bank_rows != manifest_rows or len(bank_rows) != len(region_rows):
        errors.append("Region labels/groups do not match feature bank")
    if bank["vectors"].shape != (len(region_rows), 1152) or not np.isfinite(bank["vectors"]).all():
        errors.append("Invalid region feature dimensions/values")

    def walk(value):
        if isinstance(value, list):
            for v in value:
                walk(v)
        elif isinstance(value, dict):
            for field, digest_key in (
                ("path", "sha256"),
                ("parent_path", "parent_sha256"),
                ("view_path", "view_sha256"),
                ("region_path", "region_sha256"),
            ):
                if not isinstance(value.get(field), str):
                    continue
                p = (root / value[field]).resolve()
                if not p.is_relative_to(root) or not p.is_file():
                    errors.append("Missing or external image: " + value[field])
                    continue
                expected = value.get(digest_key)
                if verify_hashes and expected:
                    if p not in seen:
                        seen[p] = hashlib.sha256(p.read_bytes()).hexdigest()
                    if seen[p] != expected:
                        errors.append("Image hash changed: " + value[field])
            for v in value.values():
                if isinstance(v, (dict, list)):
                    walk(v)

    for name in (
        "manifest.json",
        "regions.json",
        "region_control.json",
        "annotations.json",
        "scenes.json",
    ):
        walk(read(root / "datasets" / name))
    full, crop = load_features(root)
    rows = training + controls + data["condition_rows"]
    index = read(root / "datasets/features/index.json")
    for r in rows:
        if r["key"] not in full or r["key"] not in crop:
            errors.append("Missing features: " + r["key"])
        if index.get(r["key"]) != r["sha256"]:
            errors.append("Stale features: " + r["key"])
    return {
        "passed": not errors,
        "errors": errors,
        "hashed_images": len(seen),
        "train_rows": len(training),
        "core_control_rows": len(data["test"]),
        "region_rows": len(region_rows),
        "condition_overlap_note": data.get("legacy_condition_recognition_overlap", []),
    }
