"""Train reviewed MVP heads with group validation and unchanged recognition controls."""

import gc
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold

from scripts.benchmark_grouped_recognition import digest, embeddings, encoder
from scripts.benchmark_tractor_localization import crop, load_detector, tractor_box
from scripts.diagnose_round6_regression import features
from scripts.predict_tractor_conditions import CONVNEXT_WEIGHTS
from scripts.train_machinery_hierarchy import calibrate, serialize
from src.config.machinery_taxonomy import LABELS, MTZ_SPECIALIST, family_of, type_of
from src.models.hierarchy import predict_hierarchy, vote

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/mvp"


def save(name, value):
    path = OUT / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def fit(rows, x, c):
    group_counts = Counter(r["group"] for r in rows)
    y = np.array([r["model"] for r in rows])
    weights = np.array([1.0 / group_counts[r["group"]] for r in rows])
    for label in set(y):
        weights[y == label] /= weights[y == label].sum()
    weights *= len(rows) / weights.sum()
    return LogisticRegression(C=c, max_iter=3000).fit(x, y, sample_weight=weights)


def train(rows, vectors, tune=True, min_groups=8):
    x = np.stack([vectors[r["key"]] for r in rows])
    y = np.array([r["model"] for r in rows])
    groups = np.array([r["group"] for r in rows])
    classes = np.unique(y)
    folds = list(StratifiedGroupKFold(3, shuffle=True, random_state=911).split(x, y, groups))
    candidates = []
    for c in ([0.03, 0.1, 0.3, 1.0, 3.0] if tune else [1.0]):
        probabilities = np.zeros((len(rows), len(classes)))
        for a, b in folds:
            fold = fit([rows[i] for i in a], x[a], c)
            p = fold.predict_proba(x[b])
            for i, label in enumerate(fold.classes_):
                probabilities[b, list(classes).index(label)] = p[:, i]
        prediction = classes[probabilities.argmax(axis=1)]
        score = balanced_accuracy_score(y, prediction)
        candidates.append((score, c, probabilities))
    score, c, probabilities = max(candidates, key=lambda v: (v[0], -v[1]))
    model = fit(rows, x, c)
    records = [dict(group=r["group"], expected=r["model"], probabilities=p.tolist())
               for r, p in zip(rows, probabilities, strict=True)]
    thresholds = calibrate(records, classes, min_groups=min_groups, target_precision=0.95)
    return serialize(model), dict(
        c=c, grouped_oof_balanced_accuracy=score, thresholds=thresholds,
        groups=len(set(groups)), examples=len(rows),
        candidates=[dict(c=v[1], score=v[0]) for v in candidates],
        oof_records=records,
    )


def feature_cache(files):
    cache = OUT / "extra_features.npz"
    signature = hashlib.sha256("|".join(k + ":" + digest(v)
                                       for k, v in sorted(files.items())).encode()).hexdigest()
    if cache.exists():
        z = np.load(cache)
        if str(z["signature"]) == signature:
            return dict(zip(z["keys"], z["values"], strict=True))
    primary = embeddings(encoder("dinov2", ROOT / "weights/dinov2-small.safetensors"), files, True)
    gc.collect()
    secondary = embeddings(encoder("convnext", CONVNEXT_WEIGHTS), files, True)
    result = {k: np.concatenate((primary[k], secondary[k])) for k in files}
    np.savez_compressed(cache, signature=signature, keys=list(result), values=list(result.values()))
    return result


def prepare_files(data):
    rows = {r["key"]: r for r in data["condition_rows"] + data["new_review"]
            if r.get("decision", "accepted") == "accepted"}
    files = {"full/" + k: ROOT / r["path"] for k, r in rows.items()}
    crops = {}
    needed = []
    for key, r in rows.items():
        if r.get("region_label") == "attachment":
            crops[key] = ROOT / r["path"]
            continue
        path = OUT / "views" / r["sha256"] / "machine.jpg"
        crops[key] = path
        if path.exists():
            continue
        legacy = ROOT / "output/tractor_parts/views" / r["sha256"] / "tractor.jpg"
        if legacy.exists():
            crops[key] = legacy
        else:
            needed.append((key, r, path))
    if needed:
        detector = load_detector(ROOT / "weights/grounding-dino-tiny")
        for n, (key, r, destination) in enumerate(needed):
            with Image.open(ROOT / r["path"]) as handle:
                image = ImageOps.exif_transpose(handle).convert("RGB")
            found = tractor_box(image, detector, 320, 544).get("foreground")
            machine = crop(image, found["box"]) if found else image
            destination.parent.mkdir(parents=True, exist_ok=True)
            machine.save(destination, quality=95)
            if n % 10 == 0:
                print("crop", n, "/", len(needed), flush=True)
        del detector
        gc.collect()
    files.update({"crop/" + k: path for k, path in crops.items()})
    return files


def main():
    torch.set_num_threads(5)
    data = json.loads((OUT / "dataset_manifest.json").read_bytes())
    baseline = json.loads((ROOT / "output/continuation_round6/hierarchy_bundle.json").read_bytes())
    assert data["test"] == json.loads(
        (ROOT / "output/continuation_round6/hierarchy_dataset_manifest.json").read_bytes())["test"]
    for row in data["train"] + data["family_only_train"] + data["test"] + data["condition_rows"]:
        assert digest(ROOT / row["path"]) == row["sha256"], row["key"]

    files = prepare_files(data)
    extra = feature_cache(files)
    full, cropped = features()
    old = np.load(ROOT / "output/continuation_round6/features.npz")
    for k, v in zip(old["keys"], old["values"], strict=True):
        (full if str(k).startswith("full/") else cropped)[str(k)[5:]] = v
    partial = np.load(ROOT / "output/continuation_round6/hierarchy_features.npz")
    for k, v in zip(partial["keys"], partial["values"], strict=True):
        (full if str(k).startswith("full/") else cropped)[str(k)[5:]] = v
    for k, v in extra.items():
        (full if k.startswith("full/") else cropped)[k[5:]] = v
    vectors = {r["key"]: np.concatenate((full[r["key"]], cropped[r["key"]]))
               for r in data["train"] + data["family_only_train"] + data["test"]}
    train_rows = data["train"]
    partial_rows = data["family_only_train"]
    families = [dict(r, model=family_of(r["model"])) for r in train_rows]
    families += [dict(r, model=r["family"]) for r in partial_rows]
    type_rows = [dict(r, model=type_of(r["model"]) if r["model"] in LABELS else "tractor")
                 for r in train_rows]
    type_rows += [dict(r, model=r["vehicle_type"]) for r in partial_rows]
    specialist = [r for r in train_rows if r["model"] in MTZ_SPECIALIST]
    result = dict(baseline)
    result["heads"] = dict(baseline["heads"])
    result["calibration"] = dict(baseline["calibration"])
    training = {}
    for name, rows in (("type", type_rows), ("family", families), ("mtz_specialist", specialist)):
        head, summary = train(rows, vectors)
        result["heads"][name] = head
        result["calibration"][name] = summary["thresholds"]
        training[name] = summary
        print(name, "OOF", summary["grouped_oof_balanced_accuracy"], "C", summary["c"], flush=True)
    result["manifest_sha256"] = digest(OUT / "dataset_manifest.json")
    result["specialist_training"] = [
        {k: r[k] for k in ("key", "group", "model", "sha256")} for r in specialist]
    comparison = {}
    for name, bundle in (("baseline", baseline), ("candidate", result)):
        predictions = []
        for r in data["test"]:
            item = predict_hierarchy(bundle, vectors[r["key"]])
            predictions.append(dict(
                key=r["key"], expected=r["model"], expected_family=family_of(r["model"]),
                family_candidate=item["family_candidate"], family=item["family"],
                label_candidate=item["label_candidate"],
            ))
        comparison[name] = dict(
            family_correct=sum(r["family_candidate"] == r["expected_family"] for r in predictions),
            family_answers=sum(r["family"] is not None for r in predictions),
            wrong_family_answers=sum(r["family"] is not None and r["family"] != r["expected_family"]
                                     for r in predictions),
            by_class={label: dict(
                n=sum(r["expected"] == label for r in predictions),
                family_correct=sum(r["expected"] == label and r["family_candidate"] == r["expected_family"]
                                   for r in predictions),
                label_correct=sum(r["expected"] == label and r["label_candidate"] == label
                                  for r in predictions))
                for label in sorted({r["expected"] for r in predictions})},
            predictions=predictions, total=len(predictions),
        )
    # Deployment requires both family accuracy and wrong accepted answers to meet the baseline.
    better = (comparison["candidate"]["family_correct"] >= comparison["baseline"]["family_correct"]
              and comparison["candidate"]["wrong_family_answers"] <= comparison["baseline"]["wrong_family_answers"])
    save("hierarchy_candidate.json", result)
    comparison["candidate_eligible"] = better
    save("recognition_comparison.json", comparison)
    save("recognition_training.json", training)

    condition_bundle = dict(encoder_sha256=baseline["encoder_sha256"],
                            secondary_encoder_sha256=baseline["secondary_encoder_sha256"],
                            heads={}, calibration={}, training={})
    condition_metrics = {}
    for attribute in ("dirty", "snow", "rain", "wet"):
        train_rows = [dict(r, model=str(r["condition"][attribute])) for r in data["condition_rows"]
                      if r["role"] == "train" and r["condition"].get(attribute) is not None]
        positive_groups = len({r["group"] for r in train_rows if r["model"] == "1"})
        negative_groups = len({r["group"] for r in train_rows if r["model"] == "0"})
        if min(positive_groups, negative_groups) < 3:
            condition_metrics[attribute] = dict(
                enabled=False, positive_groups=positive_groups, negative_groups=negative_groups,
                reason="Insufficient independently reviewed positive/negative groups")
            continue
        # Rain uses scene context. Dirt, snow deposits and moisture use the machine crop.
        attribute_vectors = full if attribute == "rain" else cropped
        head, summary = train(train_rows, attribute_vectors, min_groups=6)
        condition_bundle["heads"][attribute] = head
        condition_bundle["calibration"][attribute] = summary["thresholds"]
        condition_bundle["training"][attribute] = summary
        test = [r for r in data["condition_rows"]
                if r["role"] == "test" and r["condition"].get(attribute) is not None]
        reports = []
        for r in test:
            label, score, _ = vote(head, attribute_vectors[r["key"]])
            threshold = summary["thresholds"][label]["threshold"]
            reports.append(dict(key=r["key"], expected=str(r["condition"][attribute]), prediction=label,
                                score=score, accepted=threshold is not None and score >= threshold))
        condition_metrics[attribute] = dict(
            enabled=True, positives=positive_groups, negatives=negative_groups,
            held_out=len(test),
            report=classification_report([r["expected"] for r in reports],
                                         [r["prediction"] for r in reports],
                                         output_dict=True, zero_division=0) if reports else {},
            accepted=sum(r["accepted"] for r in reports),
            wrong_accepted=sum(r["accepted"] and r["expected"] != r["prediction"] for r in reports),
            rows=reports,
        )
    save("condition_bundle.json", condition_bundle)
    save("condition_metrics.json", condition_metrics)

    region_rows = [dict(r, model="attachment") for r in data["region_negatives"]]
    machine_rows = [r for r in data["new_review"]
                    if r["decision"] == "accepted" and r.get("region_label") == "machine"]
    region_rows += [dict(r, model="machine") for r in machine_rows]
    region_head, region_training = train(region_rows, cropped, min_groups=5)
    save("region_bundle.json", dict(head=region_head, training=region_training,
                                   reject_threshold=0.85,
                                   encoder_sha256=baseline["encoder_sha256"],
                                   secondary_encoder_sha256=baseline["secondary_encoder_sha256"]))
    print(json.dumps({
        "recognition": {k: {m: v[m] for m in ("family_correct", "family_answers", "wrong_family_answers")}
                        for k, v in comparison.items() if isinstance(v, dict)},
        "conditions": {k: dict(enabled=v["enabled"], test=v.get("held_out"), accepted=v.get("accepted"),
                              wrong=v.get("wrong_accepted")) for k,v in condition_metrics.items()},
        "region_oof": region_training["grouped_oof_balanced_accuracy"],
    }), flush=True)


if __name__ == "__main__":
    main()
