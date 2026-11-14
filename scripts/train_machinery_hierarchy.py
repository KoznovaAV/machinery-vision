"""Train type/family heads and an independent MTZ specialist on fixed controls."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

from scripts.benchmark_grouped_recognition import digest
from scripts.benchmark_supervised_parts import fused
from scripts.benchmark_tractor_parts import fit
from scripts.diagnose_round6_regression import features
from scripts.train_field_model_pilot import assert_independent
from scripts.train_round5_recognition import evaluation_label, prepare_crops, summarize
from src.config.machinery_taxonomy import (
    LABELS,
    MTZ_SPECIALIST,
    TAXONOMY_SHA256,
    TAXONOMY_VERSION,
    family_of,
    type_of,
)
from src.models.hierarchy import predict_hierarchy, vote

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/continuation_round6"


def serialize(model):
    return dict(
        classes=model.classes_.tolist(),
        coef=model.coef_.tolist(),
        intercept=model.intercept_.tolist(),
    )


def specialist_rows(rows):
    return [dict(r) for r in rows if r["model"] in MTZ_SPECIALIST]


def calibrate(records, classes, min_groups=10, target_precision=0.95):
    by = defaultdict(list)
    for r in records:
        by[r["group"]].append(r)
    grouped = []
    for group, rr in by.items():
        truth = {r["expected"] for r in rr}
        if len(truth) != 1:
            continue
        scores = np.mean([r["probabilities"] for r in rr], axis=0)
        idx = int(scores.argmax())
        grouped.append(
            dict(prediction=str(classes[idx]), expected=next(iter(truth)), score=float(scores[idx]))
        )
    result = {}
    for label in classes:
        pred = [r for r in grouped if r["prediction"] == label]
        valid = []
        for threshold in sorted({r["score"] for r in pred}):
            selected = [r for r in pred if r["score"] >= threshold]
            correct = sum(r["expected"] == label for r in selected)
            if len(selected) >= min_groups and correct / len(selected) >= target_precision:
                valid.append(
                    dict(
                        threshold=threshold,
                        groups=len(selected),
                        correct=correct,
                        training_oof_precision=correct / len(selected),
                    )
                )
        result[str(label)] = (
            valid[0]
            if valid
            else dict(
                threshold=None,
                groups=0,
                reason="Insufficient independent OOF support at target precision",
            )
        )
    return result


def train_head(rows, vectors, calibration=False):
    x = np.stack([vectors[r["key"]] for r in rows])
    y = np.array([r["model"] for r in rows])
    groups = np.array([r["group"] for r in rows])
    model = fit(rows, x)
    head = serialize(model)
    if not calibration:
        return head, None
    records = []
    for train, valid in StratifiedGroupKFold(3, shuffle=True, random_state=609).split(x, y, groups):
        assert not (set(groups[train]) & set(groups[valid]))
        selected = [rows[i] for i in train]
        fold = fit(selected, x[train])
        prob = fold.predict_proba(x[valid])
        aligned = np.zeros((len(valid), len(model.classes_)))
        for j, label in enumerate(fold.classes_):
            aligned[:, list(model.classes_).index(label)] = prob[:, j]
        for i, scores in zip(valid, aligned, strict=True):
            records.append(
                dict(
                    group=rows[i]["group"], expected=rows[i]["model"], probabilities=scores.tolist()
                )
            )
    return head, dict(
        thresholds=calibrate(records, model.classes_),
        records=records,
        min_groups=10,
        target_precision=0.95,
        folds=3,
    )


def main():
    torch.set_num_threads(5)
    data = json.loads((OUT / "hierarchy_dataset_manifest.json").read_text(encoding="utf-8"))
    train = data["train"]
    partial = data.get("family_only_train", [])
    test = data["test"]
    assert_independent(train + partial, test)
    old = json.loads(
        (ROOT / "output/continuation_round5/dataset_manifest.json").read_text(encoding="utf-8")
    )
    assert test == old["test"]
    for r in train + partial + test:
        if digest(ROOT / r["path"]) != r["sha256"]:
            raise ValueError("Changed dataset image")
    full, cropped = features()
    oldcache = np.load(OUT / "features.npz")
    for k, v in zip(oldcache["keys"], oldcache["values"], strict=True):
        if str(k).startswith("full/"):
            full[str(k)[5:]] = v
        else:
            cropped[str(k)[5:]] = v
    missing = [r for r in train + partial if r["key"] not in full]
    import scripts.train_round5_recognition as base

    base.OUT = OUT
    paths = prepare_crops(missing)
    files = {
        **{"full/" + r["key"]: ROOT / r["path"] for r in missing},
        **{"crop/" + r["key"]: paths[r["key"]] for r in missing},
    }
    signature = "|".join(k + ":" + digest(v) for k, v in sorted(files.items()))
    cache = OUT / "hierarchy_features.npz"
    if cache.exists():
        z = np.load(cache)
        assert str(z["signature"]) == signature
        extra = dict(zip(z["keys"], z["values"], strict=True))
    else:
        extra = fused(files)
        np.savez_compressed(
            cache, signature=signature, keys=list(extra), values=list(extra.values())
        )
    for k, v in extra.items():
        if k.startswith("full/"):
            full[k[5:]] = v
        else:
            cropped[k[5:]] = v
    vectors = {
        r["key"]: np.concatenate((full[r["key"]], cropped[r["key"]]))
        for r in train + partial + test
    }
    flat, _ = train_head(train, vectors)
    families = [dict(r, model=family_of(r["model"])) for r in train] + [
        dict(r, model=r["family"]) for r in partial
    ]
    # Every negative in this audited corpus is a tractor of an unsupported model.
    types = [
        dict(r, model=type_of(r["model"]) if r["model"] in LABELS else "tractor") for r in train
    ] + [dict(r, model=r["vehicle_type"]) for r in partial]
    parent, fc = train_head(families, vectors, True)
    typehead, tc = train_head(types, vectors, True)
    child, cc = train_head(specialist_rows(train), vectors, True)
    bundle = dict(
        taxonomy_version=TAXONOMY_VERSION,
        taxonomy_sha256=TAXONOMY_SHA256,
        decision_policy="hierarchical_review_only",
        feature_encoder="fused_full_tractor",
        input=224,
        detector_size=[320, 544],
        preprocess="ImageOps.pad224, ImageNet normalize",
        heads=dict(type=typehead, family=parent, mtz_specialist=child),
        calibration=dict(
            type=tc["thresholds"], family=fc["thresholds"], mtz_specialist=cc["thresholds"]
        ),
        manifest_sha256=digest(OUT / "hierarchy_dataset_manifest.json"),
        protected_test_manifest_sha256=digest(
            ROOT / "output/continuation_round5/dataset_manifest.json"
        ),
        encoder_sha256=digest(ROOT / "weights/dinov2-small.safetensors"),
        detector_sha256=digest(ROOT / "weights/grounding-dino-tiny/model.safetensors"),
        specialist_training=[
            {k: r[k] for k in ["key", "group", "model", "sha256"]} for r in specialist_rows(train)
        ],
    )
    from scripts.predict_tractor_conditions import CONVNEXT_WEIGHTS

    bundle["secondary_encoder_sha256"] = digest(CONVNEXT_WEIGHTS)
    (OUT / "hierarchy_bundle.json").write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "training_oof.json").write_text(
        json.dumps(dict(family=fc, type=tc, mtz_specialist=cc), indent=2), encoding="utf-8"
    )
    oldruns = json.loads(
        (ROOT / "output/continuation_round5/comparison.json").read_text(encoding="utf-8")
    )["runs"]
    runs = {
        k: v for k, v in oldruns.items() if k in {"expanded_full_crop", "previous_six_full_crop"}
    }
    runs["flat_round6"] = []
    runs["hierarchy_round6"] = []
    outputs = []
    for r in test:
        vector = vectors[r["key"]]
        flatlabel = vote(flat, vector)[0]
        result = predict_hierarchy(bundle, vector)
        for name, label in [
            ("flat_round6", flatlabel),
            ("hierarchy_round6", result["label_candidate"] or "other"),
        ]:
            runs[name].append(
                dict(
                    key=r["key"],
                    group=r["group"],
                    cohort=r["cohort"],
                    expected=evaluation_label(r["model"], flat["classes"]),
                    prediction=label,
                )
            )
        outputs.append(
            dict(
                key=r["key"],
                group=r["group"],
                expected_family=family_of(r["model"]),
                expected_type=type_of(r["model"]) if r["model"] in LABELS else "tractor",
                expected_label=evaluation_label(r["model"], flat["classes"]),
                result=result,
            )
        )
    summary = {k: summarize(v) for k, v in runs.items()}
    family_correct = sum(r["result"]["family_candidate"] == r["expected_family"] for r in outputs)
    type_correct = sum(r["result"]["vehicle_type_candidate"] == r["expected_type"] for r in outputs)
    asserted = [r for r in outputs if r["result"]["family"] is not None]
    focus = [r for r in test if r["model"] in MTZ_SPECIALIST]
    oracle = sum(vote(child, vectors[r["key"]])[0] == r["model"] for r in focus)
    metrics = dict(
        type_correct=type_correct,
        family_correct=family_correct,
        total=len(test),
        family_answers=len(asserted),
        wrong_family_answers=sum(r["result"]["family"] != r["expected_family"] for r in asserted),
        exact_answers=0,
        mtz_specialist_conditional=dict(correct=oracle, total=len(focus)),
        warning=(
            "Conditional specialist score assumes correct family routing; "
            "not end-to-end accuracy."
        ),
    )
    (OUT / "comparison.json").write_text(
        json.dumps(
            dict(
                summary=summary,
                runs=runs,
                hierarchy_metrics=metrics,
                outputs=outputs,
                train=len(train),
                family_only_train=len(partial),
                test=len(test),
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUT / "flat_head.json").write_text(json.dumps(flat, indent=2), encoding="utf-8")
    print(json.dumps(dict(summary=summary, metrics=metrics), indent=2), flush=True)


if __name__ == "__main__":
    main()
