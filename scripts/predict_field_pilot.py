"""Run the uncalibrated extended field-model pilot for human review only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scripts.benchmark_grouped_recognition import digest, embeddings, encoder
from scripts.predict_tractor_conditions import CONVNEXT_WEIGHTS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HEAD = ROOT / "output/continuation_round4/field_pilot/expanded_head.json"


def review_output(logits, classes):
    logits = np.asarray(logits)
    if (
        logits.shape != (len(classes),)
        or not np.isfinite(logits).all()
        or len(set(classes)) != len(classes)
    ):
        raise ValueError("Invalid candidate logits or classes")
    order = np.argsort(logits)[::-1]
    label = classes[int(order[0])]
    return dict(
        status="manual_review",
        model=None,
        visual_candidate=label,
        review_candidate=label,
        alternatives=[classes[int(i)] for i in order[:3]],
        reason="Uncalibrated extended-class prototype; human verification required.",
    )


def validate_pilot_head(head):
    mode = head.get("feature_encoder")
    if head.get("decision_policy") != "manual_review_only" or mode not in {
        "fused",
        "fused_full_tractor",
    }:
        raise ValueError("Not a supported experimental review head")
    if mode == "fused_full_tractor" and head.get("detector_size") != [320, 544]:
        raise ValueError("Unexpected pilot detector preprocessing")
    return mode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--head", type=Path, default=DEFAULT_HEAD)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--marking", action="store_true")
    args = parser.parse_args()
    head = json.loads(args.head.read_text(encoding="utf-8"))
    mode = validate_pilot_head(head)
    if head.get("input") != 224 or head.get("preprocess") != "ImageOps.pad224, ImageNet normalize":
        raise ValueError("Unexpected pilot preprocessing")
    primary = ROOT / "weights/dinov2-small.safetensors"
    if head["encoder_sha256"] != digest(primary) or head["secondary_encoder_sha256"] != digest(
        CONVNEXT_WEIGHTS
    ):
        raise ValueError("Pilot encoder weights changed")
    torch.set_num_threads(5)
    files = {"image": args.image}
    localization = None
    if mode == "fused_full_tractor":
        import os
        import sys

        import albumentations  # noqa: F401
        from PIL import ImageOps

        from scripts.benchmark_tractor_localization import crop, load_detector, tractor_box

        sys.path.insert(0, str(Path(os.environ["TEMP"]) / "machinery-detector-deps"))
        detector = load_detector(ROOT / "weights/grounding-dino-tiny")
        with Image.open(args.image) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        localization = tractor_box(image, detector, 320, 544)
        selected = localization.get("foreground")
        from io import BytesIO

        region = crop(image, selected["box"]) if selected else image
        buffer = BytesIO()
        region.save(buffer, format="JPEG", quality=95)
        buffer.seek(0)
        with Image.open(buffer) as saved:
            files["tractor"] = saved.convert("RGB")
        del detector
    a = embeddings(encoder("dinov2", primary), files, True)
    b = embeddings(encoder("convnext", CONVNEXT_WEIGHTS), files, True)
    pieces = [a["image"], b["image"]]
    if mode == "fused_full_tractor":
        pieces.extend([a["tractor"], b["tractor"]])
    vector = np.concatenate(pieces)
    coef = np.asarray(head["coef"])
    if coef.shape != (len(head["classes"]), len(vector)):
        raise ValueError("Pilot feature dimensions changed")
    result = review_output(coef @ vector + np.asarray(head["intercept"]), head["classes"])
    if localization is not None:
        result["localization"] = localization
        result["crop_fallback_full"] = "foreground" not in localization
    if args.marking:
        from rapidocr_onnxruntime import RapidOCR

        from scripts.analyze_model_marking import inspect, revised_suggestion

        with Image.open(args.image) as image:
            evidence = inspect(image, RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1))
        result["marking"] = evidence
        result["review_candidate"] = revised_suggestion(
            result["visual_candidate"], evidence, set(head["classes"])
        )
    result.update(
        experimental=True,
        image=str(args.image.resolve()),
        sha256=digest(args.image),
        head_sha256=digest(args.head),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
