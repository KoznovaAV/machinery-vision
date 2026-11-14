"""Compare manually supervised axle/marking crops on unchanged and new sources."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from scripts.benchmark_grouped_recognition import digest, embeddings, encoder
from scripts.benchmark_tractor_conditions import CLASSES
from scripts.benchmark_tractor_localization import (
    crop,
    intersection_over_union,
    load_detector,
    tractor_box,
)
from scripts.benchmark_tractor_parts import BASE, ROOT, fit, read_rows
from scripts.predict_tractor_conditions import CONVNEXT_WEIGHTS, decide

OUT = ROOT / "output/axle_round2"
VIEWS = ("axle", "marking")


def fused(files):
    a = embeddings(encoder("dinov2", ROOT / "weights/dinov2-small.safetensors"), files, True)
    b = embeddings(encoder("convnext", CONVNEXT_WEIGHTS), files, True)
    return {k: np.concatenate((v, b[k])) for k, v in a.items()}


def prepare_new(rows):
    need = [
        r
        for r in rows
        if not (OUT / "new_views" / r["key"].split("/")[-1] / "tractor.jpg").exists()
    ]
    if not need:
        return
    import albumentations  # noqa: F401

    sys.path.insert(0, str(Path(os.environ["TEMP"]) / "machinery-detector-deps"))
    detector = load_detector(ROOT / "weights/grounding-dino-tiny")
    for r in need:
        folder = OUT / "new_views" / r["key"].split("/")[-1]
        folder.mkdir(parents=True, exist_ok=True)
        with Image.open(ROOT / r["path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        detections = tractor_box(image, detector, 320, 544)
        selected = detections.get("foreground")
        machine = crop(image, selected["box"]) if selected else image
        machine.save(folder / "tractor.jpg", quality=95)
        (folder / "whole_box.json").write_text(json.dumps(detections, indent=2), encoding="utf-8")
        print("whole crop", r["key"], flush=True)


def tractor_path(r):
    if r.get("cohort") == "new":
        return OUT / "new_views" / r["key"].split("/")[-1] / "tractor.jpg"
    return ROOT / "output/tractor_parts/views" / r["sha256"] / "tractor.jpg"


def detect_parts(rows):
    from ultralytics import YOLO

    weights = OUT / "detector_v2/weights/last.pt"
    model = YOLO(weights)
    result = {}
    images = {}
    directory = OUT / "part_views"
    directory.mkdir(exist_ok=True)
    for i, r in enumerate(rows):
        path = tractor_path(r)
        prediction = model.predict(str(path), imgsz=416, device="cpu", conf=0.15, verbose=False)[0]
        boxes = []
        available = []
        with Image.open(path) as source:
            image = source.convert("RGB")
            for box, cls, score in zip(
                prediction.boxes.xyxy.tolist(),
                prediction.boxes.cls.tolist(),
                prediction.boxes.conf.tolist(),
                strict=True,
            ):
                boxes.append(dict(label=VIEWS[int(cls)], xyxy=box, score=float(score)))
            for view in VIEWS:
                candidates = [b for b in boxes if b["label"] == view]
                selected = max(candidates, key=lambda b: b["score"]) if candidates else None
                available.append(float(selected is not None))
                if selected:
                    x1, y1, x2, y2 = selected["xyxy"]
                    pad = 0.1 * max(x2 - x1, y2 - y1)
                    region = image.crop(
                        (
                            max(0, int(x1 - pad)),
                            max(0, int(y1 - pad)),
                            min(image.width, int(x2 + pad + 1)),
                            min(image.height, int(y2 + pad + 1)),
                        )
                    )
                    filename = directory / (r["sha256"] + "-" + view + ".jpg")
                    region.save(filename, quality=95)
                    images[r["key"] + "/" + view] = filename
        result[r["key"]] = dict(
            boxes=boxes, available=available, tractor_path=str(path.relative_to(ROOT))
        )
        if (i + 1) % 50 == 0:
            print("parts", i + 1, "/", len(rows), flush=True)
    (OUT / "detections.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result, images


def evaluate_localization(detections):
    annotations = json.loads(
        (ROOT / "reports/axle_marking_annotations.json").read_text(encoding="utf-8")
    )
    records = []
    for r in annotations:
        if r["role"] == "train" or r["key"] not in detections:
            continue
        with Image.open(ROOT / r["path"]) as im:
            w, h = im.size
        for label, view in [("front_axle", "axle"), ("model_marking", "marking")]:
            gt = [b for b in r["boxes"] if b["label"] == label]
            found = [b for b in detections[r["key"]]["boxes"] if b["label"] == view]
            boxes = [[v * (w if i % 2 == 0 else h) for i, v in enumerate(b["xyxy"])] for b in gt]
            best = max(
                (intersection_over_union(a, b["xyxy"]) for a in boxes for b in found), default=0.0
            )
            records.append(
                dict(
                    key=r["key"],
                    cohort=r.get("cohort", "legacy"),
                    label=label,
                    annotated=len(gt),
                    predicted=len(found),
                    best_iou=best,
                    hit=bool(gt and best >= 0.5),
                )
            )
    return records


def main():
    torch.set_num_threads(5)
    rows, meta, head = read_rows()
    for r in rows:
        r["cohort"] = "legacy"
    new = json.loads(
        (ROOT / "reports/axle_marking_new_control.json").read_text(encoding="utf-8")
    )
    train = [r for r in rows if r["role"] == "train"]
    old_groups = {r["group"] for r in rows}
    old_sha = {r["sha256"] for r in rows}
    if any(r["group"] in old_groups or r["sha256"] in old_sha for r in new):
        raise ValueError("New control overlaps prior data")
    for r in new:
        if digest(ROOT / r["path"]) != r["sha256"]:
            raise ValueError("Changed new image")
    prepare_new(new)
    rows += new
    # Use the fixed last epoch; validation/control outcomes cannot select the checkpoint.
    detections, files = detect_parts(rows)
    part = fused(files) if files else {}
    data = np.load(BASE / "features.npz")
    full = dict(zip(data["keys"], data["values"], strict=True))
    previous = np.load(ROOT / "output/tractor_parts/features.npz")
    prev = dict(zip(previous["keys"], previous["values"], strict=True))
    machine = {r["key"]: prev[r["key"] + "/tractor"] for r in rows if r["cohort"] == "legacy"}
    if new:
        items = {r["key"] + "/full": ROOT / r["path"] for r in new}
        items.update({r["key"] + "/tractor": tractor_path(r) for r in new})
        vectors = fused(items)
        for r in new:
            full[r["key"]] = vectors[r["key"] + "/full"]
            machine[r["key"]] = vectors[r["key"] + "/tractor"]
    dim = 1152
    test = [r for r in rows if r["role"] == "test"]
    configs = {
        "full": ("full",),
        "full_tractor": ("full", "tractor"),
        "full_tractor_axle": ("full", "tractor", "axle"),
        "full_tractor_marking": ("full", "tractor", "marking"),
        "full_tractor_parts": ("full", "tractor", "axle", "marking"),
    }

    def features(r, views):
        output = []
        for view in views:
            if view == "full":
                v = full[r["key"]]
            elif view == "tractor":
                v = machine[r["key"]]
            else:
                v = part.get(r["key"] + "/" + view, np.zeros(dim, dtype=np.float32))
                output.append(np.array([float(r["key"] + "/" + view in part)]))
            output.append(v)
        return np.concatenate(output)

    records = {
        r["key"]: dict(
            key=r["key"],
            expected=r["model"],
            group=r["group"],
            cohort=r["cohort"],
            baseline=decide(full[r["key"]], head),
            detections=detections[r["key"]],
        )
        for r in test
    }
    heads = {}
    for name, views in configs.items():
        model = fit(train, np.stack([features(r, views) for r in train]))
        predicted = model.predict(np.stack([features(r, views) for r in test]))
        heads[name] = dict(
            classes=model.classes_.tolist(),
            coef=model.coef_.tolist(),
            intercept=model.intercept_.tolist(),
            views=list(views),
        )
        for r, label in zip(test, predicted, strict=True):
            records[r["key"]][name] = str(label)
    previous_records = {
        r["key"]: r
        for r in json.loads(
            (ROOT / "output/tractor_parts/comparison.json").read_text(encoding="utf-8")
        )["predictions"]
    }
    for r in records.values():
        if r["cohort"] == "legacy":
            r["previous_parts_exact"] = previous_records[r["key"]]["agreement_exact"]
        r["supervised_agreement_exact"] = bool(
            r.get("previous_parts_exact", False)
            and r["full_tractor_parts"] == r["baseline"]["suggestion"]
        )
    summary = {}
    for cohort in ("legacy", "new"):
        selected = [r for r in records.values() if r["cohort"] == cohort]
        known = [r for r in selected if r["expected"] in CLASSES]
        summary[cohort] = {}
        for name in ("baseline", *configs):

            def prediction(r):
                return r[name]["suggestion"] if name == "baseline" else r[name]

            summary[cohort][name] = dict(
                correct=sum(prediction(r) == r["expected"] for r in known),
                total=len(known),
                by_class={
                    label: dict(
                        correct=sum(
                            prediction(r) == label for r in known if r["expected"] == label
                        ),
                        total=sum(r["expected"] == label for r in known),
                    )
                    for label in sorted(CLASSES)
                },
                outside_rejected=sum(
                    prediction(r) == "other" for r in selected if r["expected"] not in CLASSES
                ),
            )
    result = dict(
        summary=summary,
        training=len(train),
        control=len(test),
        predictions=list(records.values()),
        localization=evaluate_localization(detections),
        heads=heads,
        weights_sha256=digest(OUT / "detector_v2/weights/last.pt"),
        annotations_sha256=digest(ROOT / "reports/axle_marking_annotations.json"),
        note=(
            "Exploratory fixed-epoch last.pt detector. No exact MTZ upgrades. "
            "New control never trained."
        ),
    )
    (OUT / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
