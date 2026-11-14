"""Train and compare reject options on grouped, manually reviewed photos."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = ROOT / "output/open_set/filter.json"


def unit(vectors):
    values = np.asarray(vectors, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def filter_scores(vector, spec):
    x = unit(vector)
    logit = float(x @ np.asarray(spec["coef"]) + spec["intercept"])
    return {
        "linear": 0.5 * (1 + math.tanh(logit / 2)),
        "similarity": float(np.max(np.asarray(spec["centroids"]) @ x)),
    }


def threshold_at_coverage(values, labels):
    labels = np.asarray(labels, dtype=bool)
    known, other = values[labels], values[~labels]
    if not len(known) or not len(other):
        raise ValueError("The calibration split needs both labels.")
    cutoffs = [t for t in np.unique(values) if np.mean(known >= t) >= 0.90]
    return float(max(cutoffs, key=lambda t: (np.mean(other < t), np.mean(known >= t), t)))


def fit_filter(vectors, rows):
    from sklearn.linear_model import LogisticRegression

    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    counts = Counter(r["group"] for r in train)
    x = unit([vectors[r["id"]] for r in train])
    y = np.array([r["known"] for r in train])
    weights = np.array([1 / counts[r["group"]] for r in train])
    for target in (True, False):
        weights[y == target] *= len(train) / (2 * weights[y == target].sum())

    model = LogisticRegression(C=15, max_iter=1000)
    model.fit(x, y, sample_weight=weights)
    labels = sorted({r["label"] for r in train if r["known"]})
    centroids = []
    for label in labels:
        selected = np.array([r["known"] and r["label"] == label for r in train])
        centroids.append(unit(np.average(x[selected], axis=0, weights=weights[selected])).tolist())
    spec = {
        "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "centroids": centroids,
    }
    scores = [filter_scores(vectors[r["id"]], spec) for r in val]
    targets = [r["known"] for r in val]
    spec["threshold"] = {
        name: threshold_at_coverage(np.array([s[name] for s in scores]), targets)
        for name in ("linear", "similarity")
    }
    return spec


def evaluate(vectors, rows, spec, threshold):
    result = []
    for row in rows:
        item = {k: row[k] for k in ("id", "group", "split", "known", "label")}
        scores = filter_scores(vectors[row["id"]], spec)
        scores["v3"] = row["v3"]
        item["scores"] = scores
        item["accept"] = {
            name: bool(float(score) >= threshold[name]) for name, score in scores.items()
        }
        result.append(item)
    return result


def main():
    import torch
    from PIL import Image, ImageOps

    from scripts.benchmark_grouped_recognition import digest, embeddings, encoder, read_csv
    from src.data.transforms import get_val_transforms
    from src.models.loader import load_multi_task_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ARTIFACT.parent)
    args = parser.parse_args()
    base = args.data
    family, mtz = [], []
    files = {}

    def add(rows, key, path, group, split, known, label, sha=None):
        if sha and digest(path) != sha:
            raise ValueError(f"Image has changed: {path}")
        files[key] = path
        rows.append(dict(id=key, group=group, split=split, known=known, label=label))

    for r in read_csv(ROOT / "output/dataset_audit_next/family_split.csv"):
        if r["split"] != "excluded":
            key = "family/" + r["path"]
            add(
                family,
                key,
                ROOT / "data" / r["path"],
                r["group"],
                r["split"],
                True,
                r["family"],
                r["sha256"],
            )

    expansion = ROOT / "output/mtz_expansion"
    # Validation groups for the model filter are distinct from both historical and attachment tests.
    reserved = {"tractorbg-110228", "tractorbg-149208", "tractorbg-108598", "tractorbg-63069"}
    for r in read_csv(expansion / "reviewed_manifest.csv"):
        split = {"fresh_test": "test", "stress_snow": "snow", "fresh_train": "train"}[r["subset"]]
        group = r["group"]
        if group in reserved:
            split = "val"
        add(
            mtz,
            "mtz/" + r["id"],
            expansion / r["file"],
            group,
            split,
            True,
            r["model"],
            r["sha256"],
        )

    for r in read_csv(base / "reviewed_candidates.csv"):
        common = (
            r["id"],
            base / r["file"],
            r["group"],
            r["split"],
            False,
            r["category"],
            r["sha256"],
        )
        if r["scope"] == "other_family":
            add(family, *common)
        else:
            add(mtz, *common)

    attachment_rows = read_csv(base / "manual_review.csv")
    for r in attachment_rows:
        if r["duplicate_of"]:
            continue
        if digest(base / r["file"]) != r["sha256"]:
            raise ValueError(f"Changed attachment: {r['id']}")
        files[r["id"]] = base / r["file"]
        if r["family_scope"]:
            add(
                family,
                r["id"],
                files[r["id"]],
                r["group"],
                "attachments",
                r["family_scope"] == "known",
                r["model"] or r["family_scope"],
            )
        if r["model"]:
            add(mtz, r["id"], files[r["id"]], r["group"], "attachments", True, r["model"])

    for rows in (family, mtz):
        seen = {}
        for row in rows:
            if seen.setdefault(row["group"], row["split"]) != row["split"]:
                raise ValueError(f"Group crosses splits: {row['group']}")

    torch.set_num_threads(6)
    weights = ROOT / "weights/dinov2-small.safetensors"
    dino = encoder("dinov2", weights)
    print(f"Extracting DINOv2 features from {len(files)} photos", flush=True)
    vectors = embeddings(dino, files, pad=True)
    specs = {"family": fit_filter(vectors, family), "mtz": fit_filter(vectors, mtz)}
    print(f"Calibrated filters: { {k: x['threshold'] for k, x in specs.items()} }", flush=True)

    reference = load_multi_task_model(ROOT / "weights/multi_task_best_v3.ckpt")
    transforms = get_val_transforms()
    controls = sorted(
        {r["id"] for r in family + mtz if r["split"] != "train"}
        | {r["id"] for r in attachment_rows if not r["duplicate_of"]}
    )
    baselines = {}
    print(f"Running v3 on {len(controls)} controls", flush=True)
    for start in range(0, len(controls), 12):
        keys = controls[start : start + 12]
        batch = []
        for key in keys:
            with Image.open(files[key]) as im:
                image = ImageOps.exif_transpose(im).convert("RGB")
            batch.append(transforms(image=np.asarray(image))["image"])
        with torch.inference_mode():
            logits, _ = reference(torch.stack(batch))
        baselines.update(zip(keys, logits.softmax(1).amax(1).tolist(), strict=True))

    metrics = {}
    for name, rows in (("family", family), ("mtz", mtz)):
        measured = [dict(r, v3=baselines[r["id"]]) for r in rows if r["split"] != "train"]
        thresholds = dict(specs[name]["threshold"], v3=0.60)
        metrics[name] = {
            "counts": {
                "|".join(map(str, k)): v
                for k, v in Counter((r["split"], r["known"]) for r in rows).items()
            },
            "results": evaluate(vectors, measured, specs[name], thresholds),
        }
    # Hand-marked boxes quantify the effect of cropping without assuming a detector exists.
    crops = {}
    for r in attachment_rows:
        if r["duplicate_of"]:
            continue
        with Image.open(files[r["id"]]) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            x1, y1, x2, y2 = map(int, r["crop_box"].split(","))
            pad = round(0.06 * max(x2 - x1, y2 - y1))
            crops[r["id"]] = im.crop(
                (
                    max(x1 - pad, 0),
                    max(y1 - pad, 0),
                    min(x2 + pad, im.width),
                    min(y2 + pad, im.height),
                )
            )
    crop_features = embeddings(dino, crops, pad=True)
    mtz_head = json.loads((expansion / "expanded.json").read_text(encoding="utf-8"))
    for r in attachment_rows:
        key = r["id"]
        if key in crops:
            record = {"id": key, "v3_family_score": baselines[key]}
            for name, vector in (("full", vectors[key]), ("crop", crop_features[key])):
                logit = float(vector @ np.asarray(mtz_head["coef"]) + mtz_head["intercept"])
                record[name] = {
                    "p82": 0.5 * (1 + math.tanh(logit / 2)),
                    "family": filter_scores(vector, specs["family"]),
                    "mtz": filter_scores(vector, specs["mtz"]),
                }
            metrics.setdefault("crop", []).append(record)

    artifact = {"version": 1, "encoder_sha256": digest(weights), "filters": specs}
    (base / "benchmark.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (base / "filter.json").write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print("Saved filter and benchmark", flush=True)


if __name__ == "__main__":
    main()
