"""Reproduce head and condition metrics from the dataset feature cache, without OCR."""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from src.config.runtime import ROOT
from src.data.manifest import CONTROLS, load_dataset, load_features, read
from src.training.candidates import condition_benchmark, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = (
        args.output
        or ROOT / "runs/benchmark" / datetime.now().strftime("%Y%m%d-%H%M%S")
    ).resolve()
    if not output.is_relative_to(ROOT.resolve()):
        parser.error("Output must stay in the project")
    data = load_dataset()
    full, crop = load_features()
    vectors = {
        row["key"]: np.concatenate((full[row["key"]], crop[row["key"]]))
        for split in CONTROLS
        for row in data[split]
    }
    bundle = read(ROOT / "artifacts/mvp/hierarchy_bundle.json")
    classification = {
        split: evaluate(bundle, data[split], vectors) for split in CONTROLS
    }
    for value in classification.values():
        value["protocol"] = (
            "saved_features_without_ocr_without_detector_or_scene_refusal"
        )
    state = condition_benchmark(
        read(ROOT / "artifacts/mvp/condition_bundle.json"),
        [row for row in data["condition_rows"] if row["role"] == "test"],
        full,
        crop,
    )
    inputs = [
        "datasets/manifest.json",
        "datasets/features/recognition.npz",
        "datasets/features/index.json",
        "artifacts/mvp/hierarchy_bundle.json",
        "artifacts/mvp/condition_bundle.json",
        "configs/machinery_taxonomy.json",
    ]
    protocol = {
        "classification": "Cached full/crop embeddings; no OCR, detector or scene refusals.",
        "state": "Cached embeddings and source pixels; no scene visibility refusals.",
        "independent_accuracy_estimate": False,
        "input_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in inputs
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("classification", classification),
        ("state", state),
        ("protocol", protocol),
    ):
        (output / (name + ".json")).write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8"
        )
    print(
        json.dumps(
            {
                "output": str(output),
                "test": {
                    k: classification["test"][k]
                    for k in ("total", "correct_answers", "wrong_answers", "answers")
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
