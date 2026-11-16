"""Compare MTZ heads on a frozen, visually reviewed and grouped expansion."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression

from scripts.benchmark_grouped_recognition import (
    MTZ,
    classification,
    digest,
    embeddings,
    encoder,
    read_csv,
)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_data(root, expansion):
    train, test, files = [], [], {}
    historical = read_csv(root / "output/dataset_audit/mtz_pilot_candidates.csv")
    metadata = load_json(root / "output/photo_pilot/downloaded_metadata.json")
    used = ("commons-59356347", "commons-91886717", "commons-106142809")
    for r in historical:
        if r["pilot_status"] == "candidate":
            key = r["image"]
            files[key] = root / "data" / key
            train.append(
                dict(
                    id=key,
                    model=r["historical_model_candidate"],
                    group=r["duplicate_group"],
                    subset="historical_train",
                )
            )
    archive = root / "output/external_validation"
    for r in read_csv(archive / "manifest.csv"):
        if r["status"] == "included" and r["model"]:
            files[r["id"]] = archive / r["path"]
            test.append(
                dict(id=r["id"], group=r["group"], model=r["model"], subset="historical_probe")
            )
    for r in metadata:
        if r["id"] in (*used, "commons-6920362", "commons-34522938"):
            key = r["id"]
            file = root / "output/photo_pilot/collection" / (key + ".jpg")
            if digest(file) != r["sha256"]:
                raise ValueError(f"Pilot image hash mismatch: {file}")
            files[key] = file
            subset = "pilot_train" if key in used else "historical_probe"
            (train if key in used else test).append(
                dict(id=key, group=key, model=r["model"], subset=subset)
            )
    images_root = (expansion / "images").resolve()
    expansion_rows = read_csv(expansion / "reviewed_manifest.csv")
    for r in expansion_rows:
        file = (expansion / r["file"]).resolve()
        r["id"] = "fresh/" + r["id"]
        if (
            not file.is_relative_to(images_root)
            or r["id"] in files
            or r["model"] not in MTZ
            or r["subset"] not in {"fresh_train", "fresh_test", "stress_snow"}
            or not r["group"]
            or digest(file) != r["sha256"]
        ):
            raise ValueError(f"Invalid, duplicate or changed manifest row: {r['id']}")
        files[r["id"]] = file
        (train if r["subset"] == "fresh_train" else test).append(r)
    train_groups = {(r["group"]) for r in train}
    test_groups = {(r["group"]) for r in test}
    if train_groups & test_groups:
        raise ValueError("Overlapping training and evaluation groups")
    if len({r["sha256"] for r in expansion_rows}) != len(expansion_rows):
        raise ValueError("Duplicate files inside new collection")
    return train, test, files


def score(rows, probabilities, classes):
    y = np.array([r["model"] for r in rows])
    pred = classes[probabilities.argmax(axis=1)]
    out = (
        classification(y, pred, MTZ)
        if len(set(y)) == len(MTZ)
        else {"correct": int(np.sum(y == pred)), "n": len(y), "macro_f1": None}
    )
    counts = Counter(r["group"] for r in rows)
    correct = Counter(r["group"] for r, hit in zip(rows, y == pred, strict=True) if hit)
    confident = probabilities.max(axis=1) >= 0.7
    out.update(
        groups=len(counts),
        group_accuracy=sum(correct[g] / n for g, n in counts.items()) / len(counts),
        groups_all_correct=sum(correct[g] == n for g, n in counts.items()),
        threshold_0_70={
            "accepted": int(confident.sum()),
            "correct": int((confident & (y == pred)).sum()),
        },
        results=[
            {
                "id": r["id"],
                "group": r["group"],
                "domain": r.get("domain"),
                "view": r.get("view"),
                "expected": r["model"],
                "prediction": guess,
                "confidence": float(prob.max()),
            }
            for r, guess, prob in zip(rows, pred, probabilities, strict=True)
        ],
    )
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expansion", type=Path, default=Path("output/mtz_expansion"))
    args = p.parse_args()
    root = Path.cwd()
    train, test, files = load_data(root, args.expansion)
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    dino = root / "weights/dinov2-small.safetensors"
    encoders = {
        "dinov2": ("dinov2", True, dino),
        "convnext": ("convnext", False, None),
        "dino_crop": ("dinov2", False, dino),
        "dino_gray": ("dinov2", True, dino),
    }
    configurations = [
        ("baseline", "dinov2", {"historical_train"}, False, 1),
        ("pilot_weighted", "dinov2", {"historical_train", "pilot_train"}, False, 3),
        ("expanded", "dinov2", {"historical_train", "pilot_train", "fresh_train"}, False, 1),
        (
            "expanded_group_balanced",
            "dinov2",
            {"historical_train", "pilot_train", "fresh_train"},
            True,
            1,
        ),
        ("fresh_only", "dinov2", {"fresh_train"}, True, 1),
        (
            "dino_crop_expanded",
            "dino_crop",
            {"historical_train", "pilot_train", "fresh_train"},
            True,
            1,
        ),
        (
            "dino_gray_expanded",
            "dino_gray",
            {"historical_train", "pilot_train", "fresh_train"},
            True,
            1,
        ),
        (
            "convnext_expanded",
            "convnext",
            {"historical_train", "pilot_train", "fresh_train"},
            True,
            1,
        ),
    ]
    grouped_test = {
        s: [r for r in test if r["subset"] == s] for s in sorted({r["subset"] for r in test})
    }
    result = {
        "train_by_subset": dict(Counter(r["subset"] for r in train)),
        "manifest_sha256": digest(args.expansion / "reviewed_manifest.csv"),
        "variants": {},
    }
    for name, (backbone, pad, weights) in encoders.items():
        inputs = {}
        if name == "dino_gray":
            for key, path in files.items():
                with Image.open(path) as image:
                    thumb = ImageOps.grayscale(ImageOps.exif_transpose(image)).convert("RGB")
                    thumb.thumbnail((448, 448), Image.Resampling.LANCZOS)
                    inputs[key] = thumb
        features = embeddings(encoder(backbone, weights), inputs or files, pad=pad)
        for run, arch, sources, balance_groups, pilot_weight in configurations:
            if arch != name:
                continue
            rows = [r for r in train if r["subset"] in sources]
            counts = Counter(r["group"] for r in rows)
            y = [r["model"] for r in rows]
            w = np.array(
                [
                    ((1 / counts[r["group"]]) if balance_groups else 1)
                    * (pilot_weight if r["subset"] == "pilot_train" else 1)
                    for r in rows
                ]
            )
            if balance_groups:
                w /= w.mean()
                for c in MTZ:
                    selected = np.array(y) == c
                    w[selected] *= len(rows) / (2 * w[selected].sum())
            model = LogisticRegression(
                C=1, max_iter=1500, class_weight=None if balance_groups else "balanced"
            )
            model.fit(np.stack([features[r["id"]] for r in rows]), y, sample_weight=w)
            result["variants"][run] = entry = {
                "train_images": len(rows),
                "train_groups": len(counts),
                "train_per_class": dict(Counter(y)),
                "eval": {},
            }
            for s, probe in grouped_test.items():
                probabilities = model.predict_proba(np.stack([features[r["id"]] for r in probe]))
                entry["eval"][s] = score(probe, probabilities, model.classes_)
                x = entry["eval"][s]
                print(f"{run:25} {s:18} {x['correct']}/{x['n']}  F1 {x['macro_f1']}", flush=True)
            if name == "dinov2" and run in {"expanded", "expanded_group_balanced"}:
                data = {
                    "classes": list(model.classes_),
                    "coef": model.coef_[0].tolist(),
                    "intercept": float(model.intercept_[0]),
                    "input": 224,
                    "encoder": name,
                    "encoder_sha256": digest(dino),
                    "preprocess": "letterbox, ImageNet normalize",
                }
                (args.expansion / f"{run}.json").write_text(
                    json.dumps(data) + "\n", encoding="utf-8"
                )
            (args.expansion / "benchmark.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )


if __name__ == "__main__":
    main()
