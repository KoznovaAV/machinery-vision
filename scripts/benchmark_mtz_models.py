"""Measure the effect of adding MTZ-80 and MTZ-820 to a local model classifier."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from scripts.benchmark_grouped_recognition import digest, embeddings, encoder, read_csv

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/mtz_model_pilot"
EXISTING = {"mtz_82", "mtz_1221"}
ADDED = {"mtz_80", "mtz_820"}


def add(rows, row, path, label, role, origin):
    if digest(path) != row["sha256"]:
        raise ValueError(f"Image hash mismatch: {path}")
    rows.append(
        dict(
            key=f"{origin}/{row['id']}",
            group=row["group"],
            model=label,
            split=role,
            path=path,
            sha256=row["sha256"],
        )
    )


def load_rows(manifests):
    rows = []
    expansion = ROOT / "output/mtz_expansion"
    roles = {"fresh_train": "train", "fresh_test": "test", "stress_snow": "snow"}
    for r in read_csv(expansion / "reviewed_manifest.csv"):
        add(rows, r, expansion / r["file"], r["model"], roles[r["subset"]], "expansion")
    for manifest in manifests:
        for r in read_csv(manifest):
            add(rows, r, ROOT / r["path"], r["model"], r["split"], manifest.parent.name)

    groups = {}
    for row in rows:
        group = row["group"]
        identity = (row["split"], row["model"])
        if groups.setdefault(group, identity) != identity:
            raise ValueError(f"Group crosses classes or partitions: {group}")
    if len({r["sha256"] for r in rows}) != len(rows):
        raise ValueError("Duplicate images in the experiment.")
    return rows


def add_historical(rows):
    historical = ROOT / "output/dataset_audit/mtz_pilot_candidates.csv"
    for row in read_csv(historical):
        if row["pilot_status"] != "candidate":
            continue
        path = ROOT / "data" / row["image"]
        rows.append(
            dict(
                key="historical/" + row["image"],
                group="historical/" + row["duplicate_group"],
                model=row["historical_model_candidate"],
                split="historical",
                path=path,
                sha256=digest(path),
            )
        )
    pilot = ROOT / "output/photo_pilot"
    used_ids = {"commons-59356347", "commons-91886717", "commons-106142809"}
    for row in json.loads((pilot / "downloaded_metadata.json").read_text(encoding="utf-8")):
        if row["id"] in used_ids:
            labelled = {**row, "group": row["id"]}
            add(
                rows,
                labelled,
                pilot / "collection" / (row["id"] + ".jpg"),
                row["model"],
                "historical",
                "first_pilot",
            )


def fit(rows, vectors, names, include_history, include_wave=False):
    train_splits = {"train"}
    if include_history:
        train_splits.add("historical")
    if include_wave:
        train_splits.add("train_wave")
    train = [r for r in rows if r["split"] in train_splits and r["model"] in names]
    labels = np.array([r["model"] for r in train])
    if set(labels) != names:
        raise ValueError("Training needs at least one image for each requested class.")
    per_group = Counter(r["group"] for r in train)
    weights = np.array([1 / per_group[r["group"]] for r in train])
    for name in names:
        use = labels == name
        weights[use] *= len(train) / (len(names) * weights[use].sum())
    model = LogisticRegression(C=1, max_iter=1500)
    model.fit(np.stack([vectors[r["key"]] for r in train]), labels, sample_weight=weights)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extra", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = load_rows([OUTPUT / "reviewed.csv", *args.extra])
    add_historical(rows)
    test_hashes = {
        r["sha256"]
        for r in rows
        if r["split"] in {"test", "test_wave", "outside", "outside_wave", "snow"}
    }
    if any(r["sha256"] in test_hashes for r in rows if r["split"] == "historical"):
        raise ValueError("Historical training image overlaps evaluation.")
    torch.set_num_threads(6)
    checkpoint = ROOT / "weights/dinov2-small.safetensors"
    vectors = embeddings(
        encoder("dinov2", checkpoint), {r["key"]: r["path"] for r in rows}, pad=True
    )
    test = [
        r for r in rows if r["split"] in {"test", "test_wave", "outside", "outside_wave", "snow"}
    ]
    x = np.stack([vectors[r["key"]] for r in test])
    expanded_labels = {r["model"] for r in rows if r["split"] == "train_wave"}
    labels = sorted(EXISTING | ADDED | expanded_labels)
    payload = {
        "labels": labels,
        "partitions": [{**r, "path": str(r["path"].relative_to(ROOT))} for r in rows],
        "results": {},
    }
    runs = [
        ("two_models", EXISTING, False, False),
        ("four_models", EXISTING | ADDED, False, False),
        ("two_models_with_history", EXISTING, True, False),
        ("four_models_with_history", EXISTING | ADDED, True, False),
    ]
    if args.extra:
        runs.extend(
            [
                ("mtz_after_collection", EXISTING | ADDED, True, True),
                ("all_series", set(labels), True, True),
            ]
        )
    for name, classes, include_history, include_wave in runs:
        model = fit(rows, vectors, classes, include_history, include_wave)
        predictions = []
        for r, prob in zip(test, model.predict_proba(x), strict=True):
            best = int(np.argmax(prob))
            predictions.append(
                dict(
                    key=r["key"],
                    group=r["group"],
                    expected=r["model"],
                    split=r["split"],
                    prediction=model.classes_[best],
                    score=float(prob[best]),
                )
            )
        ranked = [r for r in predictions if r["split"] in {"test", "test_wave"}]
        payload["results"][name] = {
            "matrix": confusion_matrix(
                [r["expected"] for r in ranked],
                [r["prediction"] for r in ranked],
                labels=labels,
            ).tolist(),
            "predictions": predictions,
        }
        if len(classes) >= 4:
            candidate = {
                "classes": model.classes_.tolist(),
                "coef": model.coef_.tolist(),
                "intercept": model.intercept_.tolist(),
                "encoder": "dinov2",
                "encoder_sha256": digest(checkpoint),
                "input": 224,
                "preprocess": "letterbox, ImageNet normalize",
            }
            (args.output / f"{name}.json").write_text(
                json.dumps(candidate) + "\n", encoding="utf-8"
            )

    (args.output / "benchmark.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
