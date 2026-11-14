"""Repeat frozen-encoder training on reviewed round5 sources and unchanged controls."""

import gc
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import albumentations  # noqa: F401
import numpy as np
import torch
from PIL import Image, ImageOps

from scripts.benchmark_grouped_recognition import digest
from scripts.benchmark_supervised_parts import fused
from scripts.benchmark_tractor_localization import crop, load_detector, tractor_box
from scripts.benchmark_tractor_parts import fit
from scripts.predict_tractor_conditions import CONVNEXT_WEIGHTS
from scripts.train_field_model_pilot import assert_independent, head_prediction

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/continuation_round5"


def prepare_crops(rows):
    paths = {}
    needed = []
    for r in rows:
        old = ROOT / "output/tractor_parts/views" / r["sha256"] / "tractor.jpg"
        new = OUT / "views" / r["sha256"] / "tractor.jpg"
        paths[r["key"]] = old if old.exists() else new
        if not paths[r["key"]].exists():
            needed.append(r)
    if needed:
        sys.path.insert(0, str(Path(os.environ["TEMP"]) / "machinery-detector-deps"))
        model = load_detector(ROOT / "weights/grounding-dino-tiny")
        for i, r in enumerate(needed):
            with Image.open(ROOT / r["path"]) as source:
                im = ImageOps.exif_transpose(source).convert("RGB")
            detections = tractor_box(im, model, 320, 544)
            selected = detections.get("foreground")
            machine = crop(im, selected["box"]) if selected else im
            path = paths[r["key"]]
            path.parent.mkdir(parents=True, exist_ok=True)
            machine.save(path, quality=95)
            (path.parent / "regions.json").write_text(
                json.dumps(
                    dict(sha256=r["sha256"], detector_size=[320, 544], detections=detections),
                    indent=2,
                ),
                encoding="utf-8",
            )
            print("crop", i + 1, "/", len(needed), r["key"], flush=True)
        del model
        gc.collect()
    return paths


def evaluation_label(model, supported):
    return model if model in supported else "other"


def summarize(records):
    summary = {}
    for cohort in sorted({r["cohort"] for r in records}):
        selected = [r for r in records if r["cohort"] == cohort]
        groups = defaultdict(list)
        for r in selected:
            groups[r["group"]].append(r["prediction"] == r["expected"])
        summary[cohort] = dict(
            correct=sum(r["prediction"] == r["expected"] for r in selected),
            total=len(selected),
            groups=len(groups),
            group_mean_accuracy=float(np.mean([np.mean(v) for v in groups.values()])),
        )
    summary["by_class"] = {
        label: dict(
            correct=sum(r["prediction"] == label for r in records if r["expected"] == label),
            total=sum(r["expected"] == label for r in records),
        )
        for label in sorted({r["expected"] for r in records})
    }
    return summary


def main():
    torch.set_num_threads(5)
    data = json.loads((OUT / "dataset_manifest.json").read_text(encoding="utf-8"))
    train = data["train"]
    test = data["test"]
    rows = train + test
    assert_independent(train, test)
    for r in rows:
        if digest(ROOT / r["path"]) != r["sha256"]:
            raise ValueError("Changed image " + r["key"])
    oldhead = json.loads(
        (ROOT / "output/continuation_round4/field_pilot/expanded_head.json").read_text(
            encoding="utf-8"
        )
    )
    if oldhead["encoder_sha256"] != digest(ROOT / "weights/dinov2-small.safetensors") or oldhead[
        "secondary_encoder_sha256"
    ] != digest(CONVNEXT_WEIGHTS):
        raise ValueError("Changed encoder")
    full = {}
    for path in [
        ROOT / "output/targeted_weather2/fused/features.npz",
        ROOT / "output/continuation_round4/field_pilot/features.npz",
    ]:
        z = np.load(path)
        full.update(dict(zip(z["keys"], z["values"], strict=True)))
    crops = prepare_crops(rows)
    z = np.load(ROOT / "output/tractor_parts/features.npz")
    machine = {
        str(k).removesuffix("/tractor"): v
        for k, v in zip(z["keys"], z["values"], strict=True)
        if str(k).endswith("/tractor")
    }
    files = {}
    for r in rows:
        if r["key"] not in full:
            files["full/" + r["key"]] = ROOT / r["path"]
        if r["key"] not in machine:
            files["crop/" + r["key"]] = crops[r["key"]]
    signature = "|".join(
        [
            oldhead["encoder_sha256"],
            oldhead["secondary_encoder_sha256"],
            *(k + ":" + digest(v) for k, v in sorted(files.items())),
        ]
    )
    path = OUT / "features.npz"
    if path.exists():
        z = np.load(path)
        if str(z["signature"]) != signature:
            raise ValueError("Changed round5 feature cache")
        extra = dict(zip(z["keys"], z["values"], strict=True))
    else:
        print("extracting", len(files), "views", flush=True)
        extra = fused(files)
        np.savez_compressed(
            path, signature=signature, keys=list(extra), values=list(extra.values())
        )
    for k, v in extra.items():
        if k.startswith("full/"):
            full[k[5:]] = v
        else:
            machine[k[5:]] = v
    matrices = {
        "full": full,
        "full_crop": {r["key"]: np.concatenate((full[r["key"]], machine[r["key"]])) for r in rows},
    }
    heads = {"previous_full": oldhead}
    runs = {}
    for name, vectors in matrices.items():
        model = fit(train, np.stack([vectors[r["key"]] for r in train]))
        coef = model.coef_.tolist()
        head = dict(
            experimental=True,
            decision_policy="manual_review_only",
            classes=model.classes_.tolist(),
            coef=coef,
            intercept=model.intercept_.tolist(),
            feature_encoder="fused" if name == "full" else "fused_full_tractor",
            input=224,
            preprocess="ImageOps.pad224, ImageNet normalize",
            encoder_sha256=oldhead["encoder_sha256"],
            secondary_encoder_sha256=oldhead["secondary_encoder_sha256"],
            training=[{k: r[k] for k in ["key", "group", "model", "sha256"]} for r in train],
            dataset_sha256=digest(OUT / "dataset_manifest.json"),
            detector_size=[320, 544] if name == "full_crop" else None,
        )
        heads["expanded_" + name] = head
        (OUT / ("expanded_" + name + "_head.json")).write_text(
            json.dumps(head, indent=2), encoding="utf-8"
        )
    for name, head in heads.items():
        vectors = matrices["full_crop" if name.endswith("full_crop") else "full"]
        runs[name] = [
            dict(
                key=r["key"],
                sha256=r["sha256"],
                group=r["group"],
                cohort=r["cohort"],
                condition=r.get("condition", "unlabelled_old"),
                expected=evaluation_label(r["model"], head["classes"]),
                expected_original=r["model"],
                prediction=head_prediction(head, vectors[r["key"]]),
            )
            for r in test
        ]
    oldcrop = json.loads((ROOT / "output/axle_round2/comparison.json").read_text(encoding="utf-8"))[
        "heads"
    ]["full_tractor"]
    runs["previous_six_full_crop"] = [
        dict(
            key=r["key"],
            sha256=r["sha256"],
            group=r["group"],
            cohort=r["cohort"],
            condition=r.get("condition", "unlabelled_old"),
            expected=r["model"] if r["model"] in oldcrop["classes"] else "other",
            prediction=head_prediction(oldcrop, matrices["full_crop"][r["key"]]),
        )
        for r in test
    ]
    result = dict(
        summary={k: summarize(v) for k, v in runs.items()},
        runs=runs,
        train=len(train),
        test=len(test),
        previous_head_sha256=digest(
            ROOT / "output/continuation_round4/field_pilot/expanded_head.json"
        ),
        manifest_sha256=digest(OUT / "dataset_manifest.json"),
        protocol=(
            "Fixed C=1, class/source balancing; frozen encoders; "
            "no test-based parameter or epoch selection; candidate-only outputs."
        ),
    )
    (OUT / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
