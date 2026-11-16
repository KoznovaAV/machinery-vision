"""Frozen object-localization preprocessing and context crop."""

import math
import torch

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
