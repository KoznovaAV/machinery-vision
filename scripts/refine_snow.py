"""Evaluate image and colour evidence for snow using the reviewed group splits."""

import json

import numpy as np
from PIL import Image
from sklearn.metrics import classification_report

from scripts.refine_mvp import load_vectors
from scripts.train_mvp import ROOT, OUT, save, train
from src.models.condition_features import weather_measurements
from src.models.hierarchy import vote


def main():
    data = json.loads((OUT / "dataset_manifest.json").read_bytes())
    full, _ = load_vectors()
    pixels = {}
    joined = {}
    for r in data["condition_rows"]:
        with Image.open(ROOT / r["path"]) as image:
            vector = weather_measurements(image)
        pixels[r["key"]] = vector * 5
        joined[r["key"]] = np.concatenate((full[r["key"]], vector * 5))
    rows = [dict(r, model=str(r["condition"]["snow"])) for r in data["condition_rows"]
            if r["role"] == "train"]
    candidates = {}
    for source, features in (("weather_pixels", pixels), ("scene_weather", joined)):
        head, summary = train(rows, features, min_groups=6)
        candidates[source] = (head, summary, features)
    choice = max(candidates, key=lambda k: candidates[k][1]["grouped_oof_balanced_accuracy"])
    head, summary, features = candidates[choice]
    output = json.loads((OUT / "condition_refined.json").read_bytes())
    output["heads"]["snow"] = head
    output["calibration"]["snow"] = summary["thresholds"]
    output["training"]["snow"] = summary
    output["feature_sources"]["snow"] = choice
    output["minimum_scores"] = {
        "dirty": {"0": 0.95, "1": 0.80},
        "snow": {"0": 0.95, "1": 0.95},
    }
    # Moisture has no held-out wet machine yet, so keep its result unavailable.
    output["heads"].pop("wet", None)
    held_out = [r for r in data["condition_rows"] if r["role"] == "test"]
    records = []
    for r in held_out:
        label, score, _ = vote(head, features[r["key"]])
        threshold = summary["thresholds"][label]["threshold"]
        floor = output["minimum_scores"]["snow"][label]
        records.append(dict(key=r["key"], expected=str(r["condition"]["snow"]), prediction=label,
                            score=score, accepted=threshold is not None and score >= max(threshold,floor)))
    metrics = json.loads((OUT / "condition_refined_metrics.json").read_bytes())
    metrics["snow"] = dict(
        enabled=True, source=choice, held_out=len(records),
        candidates={k:v[1]["grouped_oof_balanced_accuracy"] for k,v in candidates.items()},
        report=classification_report([r["expected"] for r in records],
                                     [r["prediction"] for r in records],
                                     output_dict=True, zero_division=0),
        accepted=sum(r["accepted"] for r in records),
        wrong_accepted=sum(r["accepted"] and r["expected"] != r["prediction"] for r in records),
        rows=records,
    )
    for row in metrics["dirty"]["rows"]:
        row["accepted"] = row["accepted"] and row["score"] >= output["minimum_scores"]["dirty"][row["prediction"]]
    metrics["dirty"]["accepted"] = sum(r["accepted"] for r in metrics["dirty"]["rows"])
    metrics["dirty"]["wrong_accepted"] = sum(r["accepted"] and r["prediction"] != r["expected"]
                                            for r in metrics["dirty"]["rows"])
    metrics["wet"]["enabled"] = False
    metrics["wet"]["reason"] = "No independent wet-positive evaluation photos"
    save("condition_release.json", output)
    save("condition_release_metrics.json", metrics)
    print({k: dict(accuracy=v.get("report",{}).get("accuracy"),
                   accepted=v.get("accepted"), wrong=v.get("wrong_accepted"),
                   enabled=v["enabled"]) for k,v in metrics.items()}, flush=True)


if __name__ == "__main__":
    main()
