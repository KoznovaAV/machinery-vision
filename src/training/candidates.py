"""Candidate training on the current dataset, without historical directory dependencies."""

import copy
import hashlib
import json
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image, ImageOps
from src.config.runtime import ROOT
from src.config.machinery_taxonomy import family_of
from src.data.manifest import load_dataset, load_features, CONTROLS, read
from src.training.grouped import train, calibrate
from src.models.hierarchy import predict_hierarchy, vote, qualified
from src.models.condition_features import weather_measurements
from src.models.mvp_inference import assess_conditions, verify_trailer

OUT = ROOT / "runs/candidate"


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def count(rr):
    return dict(
        total=len(rr),
        family_correct=sum(r["candidate"] == r["expected_family"] for r in rr),
        answers=sum(r["family"] is not None for r in rr),
        wrong_answers=sum(
            r["family"] is not None and r["family"] != r["expected_family"] for r in rr
        ),
        correct_answers=sum(r["family"] == r["expected_family"] for r in rr),
        label_correct=sum(
            r["label"] == r["expected"] for r in rr if r["expected"] not in (None, "other")
        ),
    )


def evaluate(bundle, rows, vectors, markings=None):
    rr = []
    for r in rows:
        pred = predict_hierarchy(
            bundle, vectors[r["key"]], markings.get(r["sha256"]) if markings else None
        )
        rr.append(
            dict(
                key=r["key"],
                expected=r.get("model"),
                expected_family=r.get("expected_family", family_of(r.get("model"))),
                candidate=pred["family_candidate"],
                family=pred["family"],
                label=pred["label_candidate"],
            )
        )
    return dict(
        **count(rr),
        by_class={
            str(label): count([r for r in rr if str(r["expected"]) == label])
            for label in sorted({str(r["expected"]) for r in rr})
        },
        rows=rr
    )


def train_families():
    data = load_dataset()
    baseline = read(ROOT / "artifacts/mvp/hierarchy_bundle.json")
    old_keys = set(data["base_training_keys"])
    full, crop = load_features()
    rr = data["train"] + data["family_only_train"] + [r for s in CONTROLS for r in data[s]]
    vv = {r["key"]: np.concatenate((full[r["key"]], crop[r["key"]])) for r in rr}
    families = [dict(r, model=family_of(r["model"])) for r in data["train"]] + [
        dict(r, model=r["family"]) for r in data["family_only_train"]
    ]
    known = [r for r in families if r["key"] in old_keys or r["model"] != "unknown"]
    family_head, ft = train(known, vv, tune=False, min_groups=10)
    nh_rows = [
        dict(r, model="nh_t5000_family" if r["model"] == "nh_t5000_family" else "not_nh")
        for r in families
    ]
    nh_head, nt = train(nh_rows, vv, min_groups=10)
    fc = calibrate(ft["oof_records"], family_head["classes"], min_groups=10, target_precision=0.98)
    nc = calibrate(nt["oof_records"], nh_head["classes"], min_groups=10, target_precision=0.98)
    for c, floor in ((fc, 0.90), (nc, 0.95)):
        for value in c.values():
            if value["threshold"] is not None:
                value["threshold"] = max(floor, value["threshold"])
    variants = {"baseline": baseline}
    for name in ("family_only", "nh_only", "combined"):
        b = copy.deepcopy(baseline)
        if name != "nh_only":
            b["heads"]["family"] = family_head
            b["calibration"]["family"] = fc
        if name != "family_only":
            b["heads"]["nh_specialist"] = nh_head
            b["calibration"]["nh_specialist"] = nc
        b["manifest_sha256"] = hashlib.sha256(
            (ROOT / "datasets/manifest.json").read_bytes()
        ).hexdigest()
        variants[name] = b
        save(name + "_candidate.json", b)
    comparisons = {
        name: {s: evaluate(b, data[s], vv) for s in CONTROLS} for name, b in variants.items()
    }
    gates = {}
    for name in variants:
        if name == "baseline":
            continue
        current = comparisons[name]["test"]
        old = comparisons["baseline"]["test"]
        class_gates = {
            label: dict(
                family=value["family_correct"] >= old["by_class"][label]["family_correct"],
                label=value["label_correct"] >= old["by_class"][label]["label_correct"],
                correct_answers=value["correct_answers"]
                >= old["by_class"][label]["correct_answers"],
                wrong_answers=value["wrong_answers"] <= old["by_class"][label]["wrong_answers"],
            )
            for label, value in current["by_class"].items()
        }
        controls = {
            s: dict(
                family=comparisons[name][s]["family_correct"]
                >= comparisons["baseline"][s]["family_correct"],
                wrong=comparisons[name][s]["wrong_answers"]
                <= comparisons["baseline"][s]["wrong_answers"],
            )
            for s in CONTROLS[1:]
        }
        gates[name] = dict(
            classes=class_gates,
            controls=controls,
            eligible=all(
                all(v.values()) for v in list(class_gates.values()) + list(controls.values())
            ),
        )
    save("recognition_comparison.json", comparisons)
    save("recognition_gates.json", gates)
    save("recognition_training.json", dict(family=ft, nh_specialist=nt))
    print(
        {
            name: {
                s: {
                    k: r[k]
                    for k in ("family_correct", "answers", "wrong_answers", "correct_answers")
                }
                for s, r in splits.items()
            }
            for name, splits in comparisons.items()
        },
        flush=True,
    )
    print("eligible", {k: v["eligible"] for k, v in gates.items()}, flush=True)
    print(
        "failed classes",
        {
            name: {
                label: checks
                for label, checks in value["classes"].items()
                if not all(checks.values())
            }
            for name, value in gates.items()
        },
        flush=True,
    )


def condition_benchmark(bundle, rows, full, crop):
    reports = {a: [] for a in ("dirty", "snow", "rain", "wet")}
    for r in rows:
        with Image.open(ROOT / r["path"]) as handle:
            image = ImageOps.exif_transpose(handle).convert("RGB")
        predictions = assess_conditions(full[r["key"]], crop[r["key"]], image, bundle=bundle)
        for a in reports:
            truth = r["condition"].get(a)
            if truth is None:
                continue
            pred = predictions[a]
            reports[a].append(dict(key=r["key"], group=r["group"], expected=truth, **pred))
    return {
        a: dict(
            total=len(rr),
            positive_groups=len({r["group"] for r in rr if r["expected"] == 1}),
            answers=sum(r["status"] == "assessed" for r in rr),
            wrong_answers=sum(
                r["status"] == "assessed" and r["value"] != bool(r["expected"]) for r in rr
            ),
            positive_answers=sum(r.get("value") is True for r in rr),
            correct_positive_groups=len(
                {r["group"] for r in rr if r.get("value") is True and r["expected"] == 1}
            ),
            rows=rr,
        )
        for a, rr in reports.items()
    }


def train_conditions():
    full, crop = load_features()
    data = load_dataset()
    baseline = read(ROOT / "artifacts/mvp/condition_bundle.json")
    held = [r for r in data["condition_rows"] if r["role"] == "test"]
    pixel = {}
    for r in data["condition_rows"]:
        with Image.open(ROOT / r["path"]) as h:
            im = ImageOps.exif_transpose(h).convert("RGB")
        pixel[r["key"]] = np.concatenate((full[r["key"]], weather_measurements(im) * 5))
    trials = {"baseline": baseline}
    training = {}
    for attribute in ("dirty", "snow", "rain", "wet"):
        rows = [
            dict(r, model=str(r["condition"][attribute]))
            for r in data["condition_rows"]
            if r["role"] == "train" and r["condition"].get(attribute) is not None
        ]
        groups = {
            label: len({r["group"] for r in rows if r["model"] == label}) for label in ("0", "1")
        }
        assert not {r["group"] for r in rows} & {r["group"] for r in held}
        if min(groups.values()) < 6:
            training[attribute] = dict(skipped=True, groups=groups)
            continue
        options = (
            {"crop": crop, "scene": full}
            if attribute == "dirty"
            else {"scene_weather": pixel} if attribute == "snow" else {"scene": full, "crop": crop}
        )
        candidates = {
            name: train(rows, v, min_groups=10 if attribute == "dirty" else 6)
            for name, v in options.items()
        }
        source = max(
            candidates, key=lambda name: candidates[name][1]["grouped_oof_balanced_accuracy"]
        )
        head, summary = candidates[source]
        cal = calibrate(
            summary["oof_records"],
            head["classes"],
            min_groups=10 if attribute == "dirty" else 6,
            target_precision=0.98,
        )
        for value in cal.values():
            if value["threshold"] is not None:
                value["threshold"] = max(0.95, value["threshold"])
        candidate = copy.deepcopy(baseline)
        candidate["heads"][attribute] = head
        candidate["calibration"][attribute] = cal
        candidate.setdefault("feature_sources", {})[attribute] = source
        trials[attribute] = candidate
        training[attribute] = dict(source=source, groups=groups, metrics=summary, calibration=cal)
        save(attribute + "_condition_candidate.json", candidate)
        print(attribute, groups, "source", source, "cal", cal, flush=True)
    results = {name: condition_benchmark(b, held, full, crop) for name, b in trials.items()}
    gates = {}
    for name in trials:
        if name == "baseline":
            continue
        a = results["baseline"][name]
        b = results[name][name]
        gates[name] = dict(
            no_more_errors=b["wrong_answers"] <= a["wrong_answers"],
            answers_preserved=b["answers"] - b["wrong_answers"]
            >= a["answers"] - a["wrong_answers"],
            groups_preserved=b["correct_positive_groups"] >= a["correct_positive_groups"],
        )
    save("condition_comparison.json", results)
    save("condition_training.json", training)
    save("condition_gates.json", gates)
    print(
        "RESULTS",
        {
            n: {
                a: {k: v[k] for k in ("total", "answers", "wrong_answers", "positive_answers")}
                for a, v in rr.items()
            }
            for n, rr in results.items()
        },
        flush=True,
    )
    print("GATES", gates, flush=True)


def balance(value):
    parts = [value[..., :384], value[..., 384:]]
    result = np.concatenate(
        [v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9) for v in parts], axis=-1
    )
    return result / np.maximum(np.linalg.norm(result, axis=-1, keepdims=True), 1e-9)


def choose(scores, labels, groups, floor, min_groups=5):
    choices = []
    for t in sorted({floor} | {float(v) for v in scores if np.isfinite(v) and v >= floor}):
        selected = scores >= t
        gg = set(groups[selected])
        cc = {
            g
            for g in gg
            if all(labels[i] == "attachment" for i in np.flatnonzero(selected & (groups == g)))
        }
        if len(gg) >= min_groups and len(cc) / len(gg) >= 0.98:
            choices.append(
                dict(
                    threshold=t,
                    groups=len(gg),
                    correct=len(cc),
                    training_oof_precision=len(cc) / len(gg),
                )
            )
    return choices[0] if choices else dict(threshold=None, groups=0)


def train_regions():
    z = np.load(ROOT / "datasets/features/regions.npz", allow_pickle=False)
    raw, labels, groups, keys = z["vectors"], z["labels"], z["groups"], z["keys"]
    rows = [
        {"key": str(k), "model": str(l), "group": str(g)}
        for k, l, g in zip(keys, labels, groups, strict=True)
    ]
    reference = balance(raw)
    similarity = reference @ reference.T
    margins = []
    for i in range(len(rows)):
        permitted = groups != groups[i]
        margins.append(
            float(
                similarity[i, permitted & (labels == "attachment")].max()
                - similarity[i, permitted & (labels == "machine")].max()
            )
        )
    margins = np.asarray(margins)
    reference_calibration = choose(margins, labels, groups, 0.03)
    head, metrics = train(
        rows, {r["key"]: v * 20 for r, v in zip(rows, raw, strict=True)}, tune=False, min_groups=5
    )
    probabilities = np.asarray([r["probabilities"] for r in metrics["oof_records"]])
    attachment_column = head["classes"].index("attachment")
    conditional = np.where(
        (margins >= 0.03) & (probabilities.argmax(axis=1) == attachment_column),
        probabilities[:, attachment_column],
        -np.inf,
    )
    supplement_calibration = choose(conditional, labels, groups, 0.95)
    path = OUT / "region_references.npz"
    np.savez_compressed(
        path, vectors=reference.astype("float32"), labels=labels, groups=groups, keys=keys
    )
    candidate = copy.deepcopy(read(ROOT / "artifacts/mvp/region_bundle.json"))
    config = candidate["trailer_reference"]
    config.update(
        path=path.relative_to(ROOT).as_posix(),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        normalization="per_encoder_l2",
        threshold=reference_calibration["threshold"],
        oof_groups=reference_calibration["groups"],
        oof_correct_groups=reference_calibration.get("correct", 0),
        supplement_head=head,
        supplement_scale=20,
        supplement_minimum_margin=0.03,
        training_examples=len(rows),
        training_groups=len(set(groups)),
    )
    config["supplement_calibration"]["attachment"] = supplement_calibration
    data = load_dataset()
    _, crop = load_features()
    main = [
        {"key": r["key"], **verify_trailer(crop[r["key"]], bundle=candidate)[1]}
        for name in CONTROLS
        for r in data[name]
    ]
    main_veto = [
        r["key"]
        for name in CONTROLS
        for r in data[name]
        if verify_trailer(crop[r["key"]], bundle=candidate)[0]
    ]
    held = read(ROOT / "datasets/region_control.json")
    control = np.load(ROOT / "datasets/features/region_control.npz", allow_pickle=False)
    vectors = {str(k): v for k, v in zip(control["keys"], control["values"], strict=True)}
    control_rows = [
        {
            "key": r["key"],
            "expected": r["model"],
            "veto": bool(verify_trailer(vectors[r["key"]], bundle=candidate)[0]),
        }
        for r in held
    ]
    rejected_machines = [r["key"] for r in control_rows if r["expected"] == "machine" and r["veto"]]
    save("region_candidate.json", candidate)
    save(
        "region_training.json",
        {
            "reference_calibration": reference_calibration,
            "supplement_calibration": supplement_calibration,
            "metrics": metrics,
        },
    )
    save("region_comparison.json", {"main": main, "partial_control": control_rows})
    save(
        "region_gates.json",
        {
            "main_machine_vetoes": main_veto,
            "partial_machine_vetoes": rejected_machines,
            "requires_full_scene_review": True,
            "automatic_deployment": False,
        },
    )
    print("Region candidate saved; full-scene verification is still required.", flush=True)
