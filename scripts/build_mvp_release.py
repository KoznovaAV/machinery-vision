"""Package the reviewed local MVP and persist diagnostics for the selected policies."""

import hashlib
import json

import numpy as np
from PIL import Image

from scripts.diagnose_round6_regression import features
from scripts.refine_mvp import load_vectors
from scripts.train_mvp import ROOT, OUT, save, train
from src.config.machinery_taxonomy import family_of
from src.models.condition_features import weather_measurements
from src.models.hierarchy import predict_hierarchy, vote


def main():
    data = json.loads((OUT / "dataset_manifest.json").read_bytes())
    full, cropped = features()
    for filename in ("features.npz", "hierarchy_features.npz"):
        z = np.load(ROOT / "output/continuation_round6" / filename)
        for key, value in zip(z["keys"], z["values"], strict=True):
            (full if str(key).startswith("full/") else cropped)[str(key)[5:]] = value
    extra_full, extra_crops = load_vectors()
    full.update(extra_full)
    cropped.update(extra_crops)
    baseline = json.loads((ROOT / "output/continuation_round6/hierarchy_bundle.json").read_bytes())
    candidate = json.loads((OUT / "hierarchy_fixed_c.json").read_bytes())
    candidate["manifest_sha256"] = hashlib.sha256((OUT / "dataset_manifest.json").read_bytes()).hexdigest()
    candidate["training_protocol"] = "expanded_family_original_specialist_fixed_c"
    for item in candidate["calibration"]["family"].values():
        if item["threshold"] is not None:
            item["threshold"] = max(item["threshold"], 0.90)

    def evaluate(bundle, rows):
        results = []
        for row in rows:
            vector = np.concatenate((full[row["key"]], cropped[row["key"]]))
            prediction = predict_hierarchy(bundle, vector)
            results.append(dict(
                key=row["key"], expected_family=row.get("expected_family", family_of(row.get("model"))),
                family=prediction["family"], family_candidate=prediction["family_candidate"],
            ))
        return dict(
            total=len(results),
            family_correct=sum(r["expected_family"] == r["family_candidate"] for r in results),
            family_answers=sum(r["family"] is not None for r in results),
            wrong_answers=sum(r["family"] is not None and r["family"] != r["expected_family"] for r in results),
            rows=results,
        )

    known_weather = {
        "m000": "mtz_80_family", "m002": "mtz_80_family", "m004": "mtz_80_family",
        "m006": "mtz_80_family", "m011": "mtz_80_family", "n004": "mtz_80_family",
        "n010": "mtz_80_family", "n002": "unknown", "n003": "unknown",
        "m016": "unknown", "m028": "unknown", "n012": "unknown",
    }
    family_training_hashes = {r["sha256"] for r in data["train"] + data["family_only_train"]}
    independent = [dict(r, expected_family=known_weather[r["id"]]) for r in data["new_review"]
                   if r["decision"] == "accepted" and r["id"] in known_weather
                   and r["sha256"] not in family_training_hashes]
    recognition = dict(
        baseline=evaluate(baseline, data["test"]),
        selected=evaluate(candidate, data["test"]),
        new_control_baseline=evaluate(baseline, independent),
        new_control_selected=evaluate(candidate, independent),
        note="The original control was repeatedly examined during development. The new model-control rows were excluded from family training; condition labels were used for a separate head with frozen encoders.",
    )

    condition = json.loads((OUT / "condition_release.json").read_bytes())
    condition["snow_support_required"] = True
    condition["withhold_dirt_in_snow"] = True
    metrics = json.loads((OUT / "condition_release_metrics.json").read_bytes())
    condition_rows = {r["key"]:r for r in data["condition_rows"]}
    for row in metrics["snow"]["rows"]:
        with Image.open(ROOT / condition_rows[row["key"]]["path"]) as image:
            measurements = weather_measurements(image)
        floor = condition["minimum_scores"]["snow"][row["prediction"]]
        row["accepted"] = bool(row["accepted"] and row["prediction"] == "1"
                               and row["score"] >= floor
                               and measurements[5] >= 0.15 and measurements[6] >= 0.50)
    metrics["snow"]["accepted"] = sum(r["accepted"] for r in metrics["snow"]["rows"])
    metrics["snow"]["wrong_accepted"] = sum(
        r["accepted"] and r["expected"] != r["prediction"] for r in metrics["snow"]["rows"])
    metrics["snow"]["meaning"] = "Snow visible in scene. Snow coverage on the machine has not been independently validated."
    metrics["note"] = "Repeated diagnostic evaluations informed the prototype policies. Fresh blinded validation is required."
    rows = [dict(r, model="attachment") for r in data["region_negatives"]]
    rows += [dict(r, model="machine") for r in data["new_review"]
             if r["decision"] == "accepted" and r.get("region_label") == "machine"]
    region_head, region_training = train(rows, cropped, min_groups=5)
    region = dict(
        head=region_head, training=region_training, minimum_machine_score=0.80,
        reject_threshold=0.85, encoder_sha256=baseline["encoder_sha256"],
        secondary_encoder_sha256=baseline["secondary_encoder_sha256"],
    )
    retained = []
    for row in data["test"]:
        label, score, _ = vote(region_head, cropped[row["key"]])
        retained.append(dict(key=row["key"], label=label, score=score,
                             accepted=label=="machine" and score>=0.80))
    region_control = dict(total=len(retained), retained=sum(r["accepted"] for r in retained),
                          rows=retained)
    dest = ROOT / "artifacts/mvp"
    dest.mkdir(exist_ok=True)
    outputs = {
        "hierarchy_bundle.json":candidate, "recognition_metrics.json":recognition,
        "condition_bundle.json":condition, "condition_metrics.json":metrics,
        "region_bundle.json":region, "region_control.json":region_control,
    }
    for filename, content in outputs.items():
        (dest / filename).write_text(json.dumps(content,ensure_ascii=False,indent=2),encoding="utf-8")
    save("release_manifest.json",dict(
        dataset_sha256=candidate["manifest_sha256"],
        artifacts={name:hashlib.sha256((dest/name).read_bytes()).hexdigest() for name in outputs},
        recognition={k:{field:value for field,value in v.items() if field!="rows"}
                     for k,v in recognition.items() if isinstance(v,dict)},
    ))
    print({k:{f:v[f] for f in ("total","family_correct","family_answers","wrong_answers")}
           for k,v in recognition.items() if isinstance(v,dict)},flush=True)
    print("condition", {k:dict(accepted=v.get("accepted"),wrong=v.get("wrong_accepted"),enabled=v["enabled"])
                        for k,v in metrics.items() if isinstance(v,dict)},flush=True)
    print("region",region_control["retained"],"/",region_control["total"],flush=True)


if __name__ == "__main__":
    main()
