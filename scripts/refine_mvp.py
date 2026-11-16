"""Refine state heads and compare recognition with a fixed regularization protocol."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.metrics import classification_report

from scripts.benchmark_tractor_parts import fit
from scripts.predict_tractor_conditions import CONVNEXT_WEIGHTS
from scripts.train_machinery_hierarchy import serialize
from scripts.train_mvp import OUT, ROOT, save, train
from src.config.machinery_taxonomy import LABELS, family_of
from src.models.hierarchy import predict_hierarchy, vote
from src.models.visible_condition import _load_estimator
from src.models.predict import predict_image


def load_vectors():
    z = np.load(OUT / "extra_features.npz")
    extra = dict(zip(z["keys"], z["values"], strict=True))
    return ({k[5:]: v for k, v in extra.items() if str(k).startswith("full/")},
            {k[5:]: v for k, v in extra.items() if str(k).startswith("crop/")})


def state_features(rows):
    path = OUT / "state_features.json"
    scores = json.loads(path.read_bytes()) if path.exists() else {}
    model, transform = _load_estimator()
    for i, r in enumerate(rows):
        key = r["sha256"]
        if key in scores:
            continue
        with Image.open(ROOT / r["path"]) as handle:
            im = ImageOps.exif_transpose(handle).convert("RGB")
        _, _, label, confidence = predict_image(model, im, transform)
        probability = confidence if label == 1 else 1.0 - confidence
        scores[key] = float(probability)
        if i % 20 == 0:
            print("state evidence", i, "/", len(rows), flush=True)
    save("state_features.json", scores)
    return {r["key"]: np.array([np.log(np.clip(scores[r["sha256"]], 1e-5, 1-1e-5) /
                                               np.clip(1-scores[r["sha256"]], 1e-5, 1))])
            for r in rows}


def main():
    torch.set_num_threads(5)
    data = json.loads((OUT / "dataset_manifest.json").read_bytes())
    full, crop = load_vectors()
    states = state_features(data["condition_rows"])
    baseline = json.loads((ROOT / "output/continuation_round6/hierarchy_bundle.json").read_bytes())
    output = dict(heads={}, calibration={}, training={}, feature_sources={},
                  encoder_sha256=baseline["encoder_sha256"],
                  secondary_encoder_sha256=baseline["secondary_encoder_sha256"])
    metrics = {}
    for attribute in ("dirty", "snow", "rain", "wet"):
        rows = [dict(r, model=str(r["condition"][attribute]))
                for r in data["condition_rows"]
                if r["role"] == "train" and r["condition"].get(attribute) is not None]
        pos = len({r["group"] for r in rows if r["model"] == "1"})
        neg = len({r["group"] for r in rows if r["model"] == "0"})
        if min(pos, neg) < 3:
            metrics[attribute] = dict(enabled=False, positive_groups=pos, negative_groups=neg,
                                      reason="Insufficient independent labelled groups")
            continue
        candidates = {}
        options = {"scene": full}
        if attribute == "dirty":
            options = {"crop": crop, "scene": full, "state": states}
        for name, vectors in options.items():
            head, summary = train(rows, vectors, min_groups=6)
            candidates[name] = (head, summary, vectors)
        source = max(candidates, key=lambda k: candidates[k][1]["grouped_oof_balanced_accuracy"])
        head, summary, vectors = candidates[source]
        output["heads"][attribute] = head
        output["calibration"][attribute] = summary["thresholds"]
        output["training"][attribute] = summary
        output["feature_sources"][attribute] = source
        held_out = [r for r in data["condition_rows"]
                    if r["role"] == "test" and r["condition"].get(attribute) is not None]
        records = []
        for r in held_out:
            label, score, _ = vote(head, vectors[r["key"]])
            threshold = summary["thresholds"][label]["threshold"]
            records.append(dict(key=r["key"], expected=str(r["condition"][attribute]),
                                prediction=label, score=score,
                                accepted=threshold is not None and score >= threshold))
        metrics[attribute] = dict(
            enabled=True, source=source,
            candidates={k: v[1]["grouped_oof_balanced_accuracy"] for k,v in candidates.items()},
            positive_groups=pos, negative_groups=neg, held_out=len(records),
            report=classification_report([r["expected"] for r in records],
                                         [r["prediction"] for r in records],
                                         output_dict=True, zero_division=0),
            accepted=sum(r["accepted"] for r in records),
            wrong_accepted=sum(r["accepted"] and r["expected"] != r["prediction"] for r in records),
            rows=records,
        )
        print(attribute, source, metrics[attribute]["report"]["accuracy"],
              metrics[attribute]["accepted"], metrics[attribute]["wrong_accepted"], flush=True)
    save("condition_refined.json", output)
    save("condition_refined_metrics.json", metrics)

    # Keep the original regularization while expanding the reviewed training set.
    from scripts.diagnose_round6_regression import features
    old_full, old_crop = features()
    for filename in ("features.npz", "hierarchy_features.npz"):
        z = np.load(ROOT / "output/continuation_round6" / filename)
        for k, v in zip(z["keys"], z["values"], strict=True):
            (old_full if str(k).startswith("full/") else old_crop)[str(k)[5:]] = v
    old_full.update(full)
    old_crop.update(crop)
    vectors = {r["key"]: np.concatenate((old_full[r["key"]], old_crop[r["key"]]))
               for r in data["train"] + data["family_only_train"] + data["test"]}
    family_rows = [dict(r, model=family_of(r["model"])) for r in data["train"]]
    family_rows += [dict(r, model=r["family"]) for r in data["family_only_train"]]
    head, training = train(family_rows, vectors, tune=False, min_groups=10)
    candidate = dict(baseline, heads=dict(baseline["heads"]), calibration=dict(baseline["calibration"]))
    import hashlib

    candidate["manifest_sha256"] = hashlib.sha256(
        (OUT / "dataset_manifest.json").read_bytes()
    ).hexdigest()
    candidate["training_protocol"] = "expanded_family_original_specialist_fixed_c"
    candidate["heads"]["family"] = head
    candidate["calibration"]["family"] = training["thresholds"]
    comparisons = {}
    for name, b in (("baseline", baseline), ("expanded_fixed_c", candidate)):
        records = []
        for r in data["test"]:
            pred = predict_hierarchy(b, vectors[r["key"]])
            records.append(dict(key=r["key"], expected=r["model"], family=pred["family"],
                                candidate=pred["family_candidate"], expected_family=family_of(r["model"]),
                                label=pred["label_candidate"]))
        comparisons[name] = dict(
            total=len(records),
            family_correct=sum(r["candidate"] == r["expected_family"] for r in records),
            answers=sum(r["family"] is not None for r in records),
            wrong_answers=sum(r["family"] is not None and r["family"] != r["expected_family"]
                              for r in records),
            by_class={label: dict(
                n=sum(r["expected"] == label for r in records),
                correct=sum(r["expected"] == label and r["candidate"] == r["expected_family"]
                            for r in records))
                      for label in sorted({r["expected"] for r in records})},
            rows=records,
        )
    save("hierarchy_fixed_c.json", candidate)
    save("recognition_refined_comparison.json", comparisons)
    print("recognition", {k: {n: r[n] for n in ("family_correct","answers","wrong_answers")}
                          for k,r in comparisons.items()}, flush=True)


if __name__ == "__main__":
    main()
