"""Read model markings in two image views without granting an exact identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from scripts.benchmark_tractor_parts import marking_labels

PROTOCOL = "marking-body-v1-y20-75-scale2-conf90"


def labels_in_text(text):
    labels = set(marking_labels(text))
    normalized = text.upper().translate(str.maketrans({"Т": "T", "К": "K"}))
    if re.search(r"(?<!\w)(?:ДT|DT)[- ]?75(?:[A-ZА-Я]*)?(?!\d)", normalized):
        labels.add("dt_75")
    if re.search(r"(?<!\w)T[- ]?150[- ]?K(?!\w)", normalized):
        labels.add("t_150k")
    if re.search(r"(?<!\d)1523(?:[.,]\d+)?(?!\d)", normalized):
        labels.add("mtz_1523")
    if re.search(r"(?:ACROS|АKРОС|АКРОС)[- ]*550(?!\d)", normalized):
        labels.add("acros_550")
    return sorted(labels)


def qualified_line(line):
    if line["confidence"] < 0.90:
        return []
    if re.search(r"\d\s*[:;]", line["text"]):
        return []
    return labels_in_text(line["text"])


def overlap(a, b):
    x1, y1, x2, y2 = a
    x3, y3, x4, y4 = b
    intersection = max(0, min(x2, x4) - max(x1, x3)) * max(0, min(y2, y4) - max(y1, y3))
    union = (x2 - x1) * (y2 - y1) + (x4 - x3) * (y4 - y3) - intersection
    return intersection / union if union > 0 else 0.0


def corroborate(full, body):
    """Require a compatible label and a matching physical text region in both views."""
    pairs = []
    for a in full:
        for b in body:
            common = set(qualified_line(a)) & set(qualified_line(b))
            if common and overlap(a["box"], b["box"]) >= 0.05:
                pairs.append(
                    dict(
                        labels=sorted(common),
                        full_text=a["text"],
                        body_text=b["text"],
                        full_box=a["box"],
                        body_box=b["box"],
                    )
                )
    candidates = sorted({label for p in pairs for label in p["labels"]})
    return dict(
        status=(
            "consistent_marking"
            if len(candidates) == 1
            else "ambiguous_marking" if candidates else "unconfirmed"
        ),
        suggestion=candidates[0] if len(candidates) == 1 else None,
        candidates=candidates,
        pairs=pairs,
        exact_model=None,
        warning=(
            "Two processing views share the same pixels; "
            "agreement is not identity certification."
        ),
    )


def observations(engine, image, scale=1.0, offset=(0, 0)):
    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    found, _ = engine(bgr)
    result = []
    for polygon, text, score in found or []:
        points = np.asarray(polygon)
        x = points[:, 0] / scale + offset[0]
        y = points[:, 1] / scale + offset[1]
        result.append(
            dict(
                text=text,
                confidence=float(score),
                box=[float(x.min()), float(y.min()), float(x.max()), float(y.max())],
            )
        )
    return result


def inspect(image, engine):
    image = ImageOps.exif_transpose(image).convert("RGB")
    top = round(image.height * 0.20)
    bottom = round(image.height * 0.75)
    body = image.crop((0, top, image.width, bottom))
    # Enlargement is bounded and never treated as new photographic evidence.
    scale = min(2.0, 2400 / max(body.size))
    body = body.resize(
        (max(1, round(body.width * scale)), max(1, round(body.height * scale))),
        Image.Resampling.BICUBIC,
    )
    full = observations(engine, image)
    detail = observations(engine, body, scale, (0, top))
    return dict(
        protocol=PROTOCOL,
        image_size=list(image.size),
        full=full,
        body=detail,
        **corroborate(full, detail),
    )


def revised_suggestion(baseline, evidence, supported):
    if evidence["status"] != "consistent_marking":
        return baseline
    label = evidence["suggestion"]
    return label if label in supported else "other"


def guard_decision(baseline, evidence, supported):
    """Add a reviewable hint and veto conflicts; never promote an exact answer."""
    result = dict(baseline)
    visible = evidence["suggestion"] if evidence["status"] == "consistent_marking" else None
    result["visible_marking_model"] = visible
    result["ranked_suggestion"] = revised_suggestion(baseline["suggestion"], evidence, supported)
    if visible is not None and visible != baseline["suggestion"]:
        result.update(
            status="manual_review",
            model=None,
            family=None,
            reason="Corroborated visible marking conflicts with the classifier suggestion.",
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1)
    with Image.open(args.image) as image:
        result = inspect(image, engine)
    result.update(
        path=str(args.image.resolve()), sha256=hashlib.sha256(args.image.read_bytes()).hexdigest()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
