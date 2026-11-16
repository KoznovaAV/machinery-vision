"""Run the experimental condition or MTZ classifier on local photos."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.benchmark_grouped_recognition import IMAGE_SIZE, MTZ, digest, embeddings, encoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--dino-weights", type=Path)
    parser.add_argument("image", type=Path, nargs="+")
    args = parser.parse_args()
    head = json.loads(args.head.read_text(encoding="utf-8"))
    if head["input"] != IMAGE_SIZE:
        parser.error("Unexpected model input size.")

    is_state = head["classes"] == [0, 1]
    if is_state:
        if (
            head["weights"] != "ConvNeXt-Tiny ImageNet-1K V1"
            or head["preprocess"] != "resize short edge, center crop, ImageNet normalize"
            or not 0 <= head["threshold"] <= 1
        ):
            parser.error("The condition head is incompatible with this predictor.")
    elif (
        head["classes"] != MTZ
        or head["encoder"] != "dinov2"
        or head["preprocess"] != "letterbox, ImageNet normalize"
        or args.dino_weights is None
        or digest(args.dino_weights) != head["encoder_sha256"]
    ):
        parser.error("The MTZ head requires the DINOv2 checkpoint used for training.")

    paths = {str(path): path for path in args.image}
    model = encoder("convnext" if is_state else "dinov2", args.dino_weights)
    features = embeddings(model, paths, pad=not is_state)
    threshold = head["threshold"] if is_state else 0.5
    results = []
    for path in args.image:
        score = float(features[str(path)] @ head["coef"]) + head["intercept"]
        probability = 0.5 * (1.0 + math.tanh(score / 2.0))
        if is_state:
            results.append(
                {
                    "path": str(path),
                    "state": "dirty" if probability >= threshold else "clean",
                    "p_dirty": probability,
                }
            )
        else:
            negative, positive = head["classes"]
            results.append(
                {
                    "path": str(path),
                    "model": positive if probability >= threshold else negative,
                    "scores": {negative: 1 - probability, positive: probability},
                }
            )
    print(json.dumps({"threshold": threshold, "results": results}, indent=2))


if __name__ == "__main__":
    main()
