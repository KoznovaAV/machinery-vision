"""Inspect one tractor with crop, OCR and an orientation-guided front-region branch."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from scripts.analyze_model_marking import guard_decision, inspect
from scripts.benchmark_front_axle import front_region, linear
from scripts.benchmark_grouped_recognition import digest, embeddings, encoder
from scripts.benchmark_tractor_localization import crop, load_detector, tractor_box
from scripts.benchmark_tractor_parts import BASE, OUT, ROOT, marking_candidates, marking_labels
from scripts.predict_tractor_conditions import CONVNEXT_WEIGHTS, decide, load_head


def merge_evidence(baseline, tractor, parts, candidates):
    disagreement = baseline["suggestion"] != tractor or baseline["suggestion"] != parts
    conflict = bool(
        candidates and (len(candidates) != 1 or baseline["suggestion"] not in candidates)
    )
    result = dict(baseline)
    if conflict or (baseline["status"] == "experimental_match" and disagreement):
        result.update(
            status="manual_review",
            model=None,
            family=None,
            reason="Crop/front-region disagreement or conflicting OCR evidence.",
        )
    return dict(result=result, branch_disagreement=disagreement, marking_conflict=conflict)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--front-side",
        choices=["L", "R", "center", "rear"],
        help="Human-confirmed front direction in the tractor crop",
    )
    parser.add_argument("--bundle", type=Path, default=OUT / "front_bundle.json")
    parser.add_argument("--head", type=Path, default=BASE / "expanded.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    for field, path in [
        ("encoder_sha256", ROOT / "weights/dinov2-small.safetensors"),
        ("secondary_encoder_sha256", CONVNEXT_WEIGHTS),
        ("baseline_head_sha256", args.head),
        ("crop_head_sha256", OUT / "tractor_head.npz"),
        ("detector_sha256", ROOT / "weights/grounding-dino-tiny/model.safetensors"),
    ]:
        if bundle[field] != digest(path):
            raise ValueError("Incompatible bundle: " + field)
    if bundle["preprocess"] != "ImageOps.pad224, ImageNet normalize":
        raise ValueError("Unknown preprocessing")
    head = load_head(args.head, ROOT / "weights/dinov2-small.safetensors")
    torch.set_num_threads(5)
    # The optional detector uses its isolated Transformers environment.
    import albumentations  # noqa: F401

    sys.path.insert(0, str(Path(os.environ["TEMP"]) / "machinery-detector-deps"))
    with Image.open(args.image) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    detections = tractor_box(image, load_detector(ROOT / "weights/grounding-dino-tiny"), 320, 544)
    detection = detections.get("foreground")
    tractor = crop(image, detection["box"]) if detection else image
    args.output.mkdir(parents=True, exist_ok=True)
    tractor.save(args.output / "tractor.jpg", quality=95)
    # Match the JPEG representation used by the benchmark.
    tractor = Image.open(args.output / "tractor.jpg").convert("RGB")
    first = encoder("dinov2", ROOT / "weights/dinov2-small.safetensors")
    second = encoder("convnext", CONVNEXT_WEIGHTS)

    def encode(images):
        a = embeddings(first, images, True)
        b = embeddings(second, images, True)
        return {k: np.concatenate((v, b[k])) for k, v in a.items()}

    vectors = encode({"full": image, "tractor": tractor})
    side = args.front_side or linear(bundle["orientation"], vectors["tractor"])
    front_region(tractor, side).save(args.output / "front.jpg", quality=95)
    front = encode({"front": args.output / "front.jpg"})["front"]
    front_vote = linear(bundle["heads"]["front"], front)
    combined = linear(
        bundle["heads"]["full_tractor_front"],
        np.concatenate((vectors["full"], vectors["tractor"], front)),
    )
    # The crop head is trained on exactly the same unaugmented train split.
    crop_head = np.load(OUT / "tractor_head.npz")
    crop_vote = linear(
        {k: crop_head[k].tolist() for k in ("coef", "intercept", "classes")}, vectors["tractor"]
    )
    from rapidocr_onnxruntime import RapidOCR

    marking_evidence = inspect(tractor, RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1))
    lines = []
    for line in marking_evidence["full"]:
        x1, y1, x2, y2 = line["box"]
        lines.append(
            dict(
                box=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                text=line["text"],
                confidence=line["confidence"],
                candidates=marking_labels(line["text"]),
            )
        )
    candidates = marking_candidates(lines)
    base = decide(vectors["full"], head)
    guarded = merge_evidence(base, crop_vote, combined, candidates)
    guarded["result"] = guard_decision(guarded["result"], marking_evidence, set(head["classes"]))
    result = dict(
        experimental=True,
        image=str(args.image.resolve()),
        sha256=digest(args.image),
        baseline=base,
        detections=detections,
        tractor_suggestion=crop_vote,
        front_region=dict(
            side=side,
            side_source="human" if args.front_side else "experimental_orientation_head",
            suggestion=front_vote,
            visibility="rear_view_no_confirmed_axle" if side == "rear" else "unverified",
            axle_drive_type=None,
            warning="Proposed region; rear views and attachments may hide the axle",
        ),
        combined_suggestion=combined,
        marking=dict(lines=lines, candidates=candidates, corroborated=marking_evidence),
        **guarded,
    )
    (args.output / "prediction.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
