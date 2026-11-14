"""Evaluate automatic tractor crops against full frames and hand-marked boxes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image, ImageOps

from scripts.benchmark_grouped_recognition import digest, embeddings, encoder, read_csv
from scripts.train_open_set_filter import ARTIFACT, ROOT, filter_scores
from src.config.classes import MODEL_CLASSES
from src.data.transforms import get_val_transforms
from src.models.loader import load_multi_task_model

DETECTOR = ROOT / "weights/grounding-dino-tiny"
PROMPT = "tractor. bulldozer."
MIN_SCORE = 0.25


def load_detector(path):
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        path, local_files_only=True, use_safetensors=True
    ).eval()
    return processor, model


def tractor_box(image, detector, shortest_edge=480, longest_edge=800):
    processor, model = detector
    inputs = processor(
        images=image,
        text=PROMPT,
        size={"shortest_edge": shortest_edge, "longest_edge": longest_edge},
        return_tensors="pt",
    )
    with torch.inference_mode():
        predictions = model(**inputs)
    result = processor.post_process_grounded_object_detection(
        predictions,
        inputs.input_ids,
        threshold=MIN_SCORE,
        text_threshold=0.2,
        target_sizes=[(image.height, image.width)],
    )[0]
    viable = []
    for score, raw in sorted(
        zip(result["scores"].tolist(), result["boxes"].tolist(), strict=True),
        reverse=True,
    ):
        left, top, right, bottom = raw
        box = (
            max(0, min(image.width, int(left))),
            max(0, min(image.height, int(top))),
            max(0, min(image.width, math.ceil(right))),
            max(0, min(image.height, math.ceil(bottom))),
        )
        width, height = box[2] - box[0], box[3] - box[1]
        if width >= 24 and height >= 24 and width * height >= 0.025 * image.width * image.height:
            viable.append({"box": box, "score": score, "area": width * height})
    if not viable:
        return {}
    best = viable[0]
    # Compare the default confidence ranking with a foreground hypothesis.
    plausible = [d for d in viable if d["score"] >= best["score"] * 0.75]
    largest = max(plausible, key=lambda d: d["area"])
    return {"automatic": best, "foreground": largest}


def crop(image, box):
    x1, y1, x2, y2 = box
    pad = round(0.06 * max(x2 - x1, y2 - y1))
    return image.crop(
        (
            max(0, x1 - pad),
            max(0, y1 - pad),
            min(image.width, x2 + pad),
            min(image.height, y2 + pad),
        )
    )


def intersection_over_union(a, b):
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return overlap / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap)


def load_rows(root):
    open_set = root / "output/open_set"
    mtz = root / "output/mtz_expansion"
    rows = []

    def add(row, base, subset, known=None, label=None, box=None):
        rows.append(
            dict(
                id=row["id"],
                path=base / row["file"],
                sha256=row["sha256"],
                group=row["group"],
                subset=subset,
                known=known,
                label=label,
                manual_box=box,
            )
        )

    for r in read_csv(open_set / "manual_review.csv"):
        if r["duplicate_of"]:
            continue
        known = {"known": True, "other": False}.get(r["family_scope"])
        add(
            r,
            open_set,
            "attachments",
            known,
            r["model"] or None,
            [int(v) for v in r["crop_box"].split(",")],
        )

    for r in read_csv(mtz / "reviewed_manifest.csv"):
        if r["subset"] == "fresh_test":
            add(r, mtz, "target_models", True, r["model"])

    for r in read_csv(open_set / "reviewed_candidates.csv"):
        if r["split"] == "test":
            subset = "other_models" if r["scope"] == "other_mtz" else "other_families"
            add(r, open_set, subset, subset == "other_models")

    external = root / "output/tractor_localization/reviewed_other_mtz.csv"
    if external.is_file():
        for r in read_csv(external):
            add(r, external.parent, "new_other_models", True)
    return rows


def score_batch(images, specs, head, features, reference, transforms):
    results = {}
    items = list(images.items())
    for start in range(0, len(items), 8):
        batch = items[start : start + 8]
        tensors = [transforms(image=np.asarray(im))["image"] for _, im in batch]
        with torch.inference_mode():
            logits, _ = reference(torch.stack(tensors))
            values, indices = logits.softmax(1).max(dim=1)
        for (key, _), score, idx in zip(batch, values.tolist(), indices.tolist(), strict=True):
            vector = features[key]
            family = MODEL_CLASSES[idx]
            checks = {name: filter_scores(vector, spec)["linear"] for name, spec in specs.items()}
            accept = score >= 0.60 and checks["family"] >= specs["family"]["threshold"]["linear"]
            logit = float(vector @ np.asarray(head["coef"]) + head["intercept"])
            p82 = 0.5 * (1 + math.tanh(logit / 2))
            label = "mtz_82" if p82 >= 0.5 else "mtz_1221"
            mtz_accepted = (
                accept
                and family == "mtz_belarus"
                and checks["mtz"] >= specs["mtz"]["threshold"]["linear"]
                and max(p82, 1 - p82) >= 0.65
            )
            results[key] = dict(
                family=family,
                v3_confidence=score,
                family_filter=checks["family"],
                family_accepted=accept,
                mtz_filter=checks["mtz"],
                p82=p82,
                mtz_forced=label,
                mtz_accepted=mtz_accepted,
            )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", type=Path, default=DETECTOR)
    parser.add_argument("--output", type=Path, default=ROOT / "output/tractor_localization")
    parser.add_argument(
        "--subset", action="append", help="Run only the specified evaluation subsets."
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(6)
    weights = ROOT / "weights/dinov2-small.safetensors"
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    head = json.loads((ROOT / "output/mtz_expansion/expanded.json").read_text(encoding="utf-8"))
    if {artifact["encoder_sha256"], head["encoder_sha256"]} != {digest(weights)}:
        raise ValueError("Recognition heads do not match the DINOv2 checkpoint.")
    rows = load_rows(ROOT)
    if args.subset:
        rows = [row for row in rows if row["subset"] in args.subset]
    if not rows:
        parser.error("No images match the selected subsets.")
    detector = load_detector(args.detector)
    all_views = {}
    results = []
    for row in rows:
        path = row["path"]
        if digest(path) != row["sha256"]:
            raise ValueError(f"Changed image: {path}")
        with Image.open(path) as im:
            image = ImageOps.exif_transpose(im).convert("RGB")
        start = perf_counter()
        detections = tractor_box(image, detector)
        seconds = perf_counter() - start
        identifier = row["id"]
        box = row["manual_box"]
        record = {k: v for k, v in row.items() if k != "path"}
        record.update(image_size=list(image.size), detections=detections, seconds=seconds)
        all_views[identifier + "/full"] = image
        if box:
            all_views[identifier + "/manual"] = crop(image, box)
        for name, detection in detections.items():
            detected_box = detection["box"]
            all_views[identifier + "/" + name] = crop(image, detected_box)
            if box:
                detection["iou"] = intersection_over_union(detected_box, box)
        results.append(record)
        summary = {name: detection["box"] for name, detection in detections.items()}
        print(identifier, summary, round(seconds, 2), flush=True)
    del detector
    vectors = embeddings(encoder("dinov2", weights), all_views, pad=True)
    predictions = score_batch(
        all_views,
        artifact["filters"],
        head,
        vectors,
        load_multi_task_model(ROOT / "weights/multi_task_best_v3.ckpt"),
        get_val_transforms(),
    )
    for record in results:
        identifier = record["id"]
        record["views"] = {
            name: predictions[identifier + "/" + name]
            for name in ("full", "manual", "automatic", "foreground")
            if identifier + "/" + name in predictions
        }
    payload = dict(
        model="IDEA-Research/grounding-dino-tiny",
        detector_sha256=digest(args.detector / "model.safetensors"),
        prompt=PROMPT,
        threshold=MIN_SCORE,
        images=results,
    )
    (args.output / "benchmark.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
