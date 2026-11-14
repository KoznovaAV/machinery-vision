"""Train a source-group-held-out field-machinery candidate model, never auto-accepting it."""

from __future__ import annotations

import csv
import json

import numpy as np
import torch

from scripts.benchmark_grouped_recognition import digest
from scripts.benchmark_supervised_parts import fused
from scripts.benchmark_tractor_conditions import CLASSES
from scripts.benchmark_tractor_parts import BASE, ROOT, fit, read_rows
from scripts.predict_tractor_conditions import CONVNEXT_WEIGHTS

OUT = ROOT / "output/continuation_round4/field_pilot"
FUTURE = {"dt_75", "t_150k", "mtz_1523", "acros_550"}
CATEGORY = {"DT-75": "dt_75", "T-150K": "t_150k"}


def relabel_old_negative(row, categories):
    result = dict(row)
    if result["model"] == "other":
        replacement = CATEGORY.get(categories.get(result["key"].split("/")[-1]))
        if replacement:
            result["model"] = replacement
    return result


def assert_independent(train, test):
    if {r["group"] for r in train} & {r["group"] for r in test}:
        raise ValueError("Training/control group leakage")
    if {r["sha256"] for r in train} & {r["sha256"] for r in test}:
        raise ValueError("Training/control pixel leakage")
    for split in (train, test):
        if len({r["sha256"] for r in split}) != len(split):
            raise ValueError("Repeated pixel content within split")
    if len({r["key"] for r in train + test}) != len(train + test):
        raise ValueError("Repeated dataset key")


def head_prediction(head, vector):
    return head["classes"][
        int((np.asarray(head["coef"]) @ vector + np.asarray(head["intercept"])).argmax())
    ]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(5)
    rows, meta, baseline = read_rows()
    original_train = [r for r in rows if r["role"] == "train"]
    old_test = [
        dict(r, cohort="old_supported" if r["model"] in CLASSES else "old_outside")
        for r in rows
        if r["role"] == "test"
    ]
    opened = list(
        csv.DictReader(
            (ROOT / "output/open_set/reviewed_candidates.csv").open(encoding="utf-8-sig")
        )
    )
    categories = {r["id"]: r["category"] for r in opened}
    train = [relabel_old_negative(r, categories) for r in original_train]
    registry = json.loads(
        (ROOT / "reports/future_field_model_registry.json").read_text(encoding="utf-8")
    )
    extra_train = []
    future_test = []
    for r in registry:
        record = dict(
            key="field_registry/" + r["id"],
            group=r["group"],
            model=r["model"],
            path=r["path"],
            sha256=r["sha256"],
        )
        if r["split"] == "future_train_pool":
            extra_train.append(record)
        else:
            future_test.append(dict(record, cohort="future_new_models"))
    extra_test = []
    for r in opened:
        if r["split"] == "test" and r["category"] in CATEGORY:
            extra_test.append(
                dict(
                    key="old_heldout/" + r["id"],
                    group=r["group"],
                    model=CATEGORY[r["category"]],
                    path="output/open_set/" + r["file"],
                    sha256=r["sha256"],
                    cohort="old_heldout_new_models",
                )
            )
    for index, model in [(1, "t_150k"), (11, "t_150k"), (12, "dt_75")]:
        path = ROOT / f"output/open_set/attachment_control/photo-{index:02d}.jpg"
        extra_test.append(
            dict(
                key=f"user_attachment/{index:02d}",
                group=f"user-scene-{index}",
                model=model,
                path=str(path.relative_to(ROOT)),
                sha256=digest(path),
                cohort="user_new_models",
            )
        )
    prior = json.loads((ROOT / "reports/axle_marking_new_control.json").read_text(encoding="utf-8"))
    current = json.loads(
        (ROOT / "output/continuation_round4/new_control.json").read_text(encoding="utf-8")
    )
    test = (
        old_test
        + [dict(r, cohort="prior_external") for r in prior]
        + [dict(r, cohort="round4_external") for r in current]
        + future_test
        + extra_test
    )
    assert_independent(train + extra_train, test)
    for r in train + extra_train + test:
        if digest(ROOT / r["path"]) != r["sha256"]:
            raise ValueError("Changed dataset image")
    cache = np.load(BASE / "features.npz")
    vectors = dict(zip(cache["keys"], cache["values"], strict=True))
    missing = {
        r["key"]: ROOT / r["path"] for r in train + extra_train + test if r["key"] not in vectors
    }
    signature = "|".join(
        [
            digest(ROOT / "weights/dinov2-small.safetensors"),
            digest(CONVNEXT_WEIGHTS),
            *(k + ":" + digest(v) for k, v in sorted(missing.items())),
        ]
    )
    file = OUT / "features.npz"
    if file.exists():
        z = np.load(file)
        if str(z["signature"]) != signature:
            raise ValueError("Changed pilot feature cache; use a new experiment directory")
        vectors.update(dict(zip(z["keys"], z["values"], strict=True)))
    else:
        new = fused(missing)
        vectors.update(new)
        np.savez_compressed(file, signature=signature, keys=list(new), values=list(new.values()))
    variants = {"old_negatives_relabelled": train, "expanded": train + extra_train}
    runs = {}
    heads = {}
    for name, selected in variants.items():
        model = fit(selected, np.stack([vectors[r["key"]] for r in selected]))
        predictions = model.predict(np.stack([vectors[r["key"]] for r in test]))
        runs[name] = [
            dict(
                key=r["key"],
                group=r["group"],
                cohort=r["cohort"],
                expected=r["model"] if r["model"] in CLASSES | FUTURE else "other",
                prediction=str(label),
            )
            for r, label in zip(test, predictions, strict=True)
        ]
        heads[name] = dict(
            experimental=True,
            decision_policy="manual_review_only",
            classes=model.classes_.tolist(),
            coef=model.coef_.tolist(),
            intercept=model.intercept_.tolist(),
            feature_encoder="fused",
            input=224,
            preprocess="ImageOps.pad224, ImageNet normalize",
            encoder_sha256=digest(ROOT / "weights/dinov2-small.safetensors"),
            secondary_encoder_sha256=digest(CONVNEXT_WEIGHTS),
            training=[
                dict(key=r["key"], group=r["group"], sha256=r["sha256"], model=r["model"])
                for r in selected
            ],
        )
        (OUT / (name + "_head.json")).write_text(
            json.dumps(heads[name], indent=2), encoding="utf-8"
        )
    summary = {}
    for name, records in runs.items():
        summary[name] = {}
        for co in sorted({r["cohort"] for r in records}):
            rr = [r for r in records if r["cohort"] == co]
            summary[name][co] = dict(
                correct=sum(r["prediction"] == r["expected"] for r in rr),
                total=len(rr),
                groups=len({r["group"] for r in rr}),
            )
        summary[name]["new_models_by_class"] = {
            label: dict(
                correct=sum(r["prediction"] == label for r in records if r["expected"] == label),
                total=sum(r["expected"] == label for r in records),
                groups=len({r["group"] for r in records if r["expected"] == label}),
            )
            for label in sorted(FUTURE)
        }
    result = dict(
        summary=summary,
        runs=runs,
        train=len(train + extra_train),
        test=len(test),
        relabelled_old_negatives=[
            dict(key=a["key"], old=a["model"], new=b["model"])
            for a, b in zip(original_train, train, strict=True)
            if a["model"] != b["model"]
        ],
        registry_sha256=digest(ROOT / "reports/future_field_model_registry.json"),
        note=(
            "No automatic exact answers. New-class test counts are small; "
            "old supported and unknown controls remain separate."
        ),
    )
    (OUT / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "dataset_manifest.json").write_text(
        json.dumps(dict(train=train + extra_train, test=test), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
