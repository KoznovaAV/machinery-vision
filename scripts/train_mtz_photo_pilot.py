"""Compare pilot heads using a fixed external MTZ evaluation set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scripts.benchmark_grouped_recognition import MTZ, digest, embeddings, encoder, read_csv
from sklearn.linear_model import LogisticRegression


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd()
    meta = json.loads((args.collection / "downloaded_metadata.json").read_text(encoding="utf-8"))
    role = {
        "commons-6920362": "test",
        "commons-9969336": "near_duplicate",
        "commons-59356347": "train",
        "commons-91886717": "train",
        "commons-106142809": "train",
        "commons-34522938": "test",
    }
    for row in meta:
        row["role"] = role[row["id"]]
    historical = [
        row
        for row in read_csv(root / "output/dataset_audit/mtz_pilot_candidates.csv")
        if row["pilot_status"] == "candidate"
    ]
    old = [
        row
        for row in read_csv(root / "output/external_validation/manifest.csv")
        if row["status"] == "included" and row["model"]
    ]
    group_audit = root / "output/external_validation"
    data = {row["image"]: root / "data" / row["image"] for row in historical}
    evaluation = []
    train_new = []
    for row in old:
        data[row["id"]] = group_audit / row["path"]
        evaluation.append((row["id"], row["model"], row["group"], "previous"))
    for row in meta:
        key = row["id"]
        path = args.collection / "collection" / f"{key}.jpg"
        if digest(path) != row["sha256"]:
            raise ValueError(f"Unexpected image hash for {key}.")
        if row["role"] == "near_duplicate":
            continue
        data[key] = path
        if row["role"] == "test":
            evaluation.append((key, row["model"], key, "new"))
        else:
            train_new.append((key, row["model"]))

    torch.set_num_threads(5)
    torch.set_num_interop_threads(1)
    weights = root / "weights/dinov2-small.safetensors"
    features = embeddings(encoder("dinov2", weights), data, pad=True)
    known = [(row["image"], row["historical_model_candidate"]) for row in historical]
    output = {"collection": meta, "test_frames": len(evaluation), "variants": {}}
    runs = [
        ("baseline", known, 1),
        ("with_new", known + train_new, 1),
        ("with_new_more_weight", known + train_new, 3),
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    for name, train, extra_weight in runs:
        clf = LogisticRegression(C=1, class_weight="balanced", max_iter=1500)
        clf.fit(
            np.stack([features[key] for key, _ in train]),
            [label for _, label in train],
            sample_weight=[extra_weight if key.startswith("commons-") else 1 for key, _ in train],
        )
        probs = clf.predict_proba(np.stack([features[key] for key, *_ in evaluation]))
        frames = []
        for (key, label, group, split), vector in zip(evaluation, probs, strict=True):
            best = int(vector.argmax())
            frames.append(
                {
                    "id": key,
                    "expected": label,
                    "group": group,
                    "subset": split,
                    "prediction": clf.classes_[best],
                    "score": float(vector[best]),
                }
            )
        groups = {x["group"] for x in frames}
        output["variants"][name] = {
            "frames_correct": sum(x["expected"] == x["prediction"] for x in frames),
            "groups_all_correct": sum(
                all(x["expected"] == x["prediction"] for x in frames if x["group"] == group)
                for group in groups
            ),
            "frames": frames,
        }
        if name != "baseline":
            head = {
                "classes": list(MTZ),
                "coef": clf.coef_[0].tolist(),
                "intercept": float(clf.intercept_[0]),
                "input": 224,
                "encoder": "dinov2",
                "encoder_sha256": digest(weights),
                "preprocess": "letterbox, ImageNet normalize",
            }
            filename = "mtz_experimental" if extra_weight == 1 else "mtz_experimental_weighted"
            (args.output / f"{filename}.json").write_text(json.dumps(head) + "\n", encoding="utf-8")
    (args.output / "pilot_metrics.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for name, entry in output["variants"].items():
        print(name, entry["frames_correct"], entry["groups_all_correct"], flush=True)


if __name__ == "__main__":
    main()
