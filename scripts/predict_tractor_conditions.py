"""Run the experimental tractor classifier with explicit abstention.

Example:
    python -m scripts.predict_tractor_conditions --head path/to/head.json photo.jpg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.benchmark_grouped_recognition import digest, embeddings, encoder
from scripts.benchmark_tractor_conditions import CLASSES, ROOT, exact_allowed

CONVNEXT_WEIGHTS = Path(torch.hub.get_dir()) / "checkpoints/convnext_tiny-983f1562.pth"

AMBIGUOUS = {"mtz_80", "mtz_82", "mtz_820"}


def decide(vector, head):
    """Return an exact label only when the experimental decision policy permits it."""
    labels = head["classes"]
    vector = np.asarray(vector, dtype=np.float64)
    coef = np.asarray(head["coef"], dtype=np.float64)
    refs = np.asarray(head["train_reference"], dtype=np.float64)
    ref_labels = np.asarray(head["train_reference_classes"])
    if coef.shape != (len(labels), vector.size) or refs.shape[1:] != (vector.size,):
        raise ValueError("Feature dimensions do not match the classifier.")
    logits = coef @ vector + np.asarray(head["intercept"])
    probability = np.exp(logits - logits.max())
    probability /= probability.sum()
    best = int(probability.argmax())
    suggestion = labels[best]
    mask = ref_labels == suggestion
    norm = float(np.linalg.norm(vector))
    if not norm or not mask.any():
        raise ValueError("Invalid embedding or missing class reference images.")
    similarity = float((refs[mask] @ (vector / norm)).max())
    cutoff = head["reject"]
    score = float(probability[best])
    passes = score >= cutoff["min_probability"] and similarity >= cutoff["min_similarity"]
    if suggestion == "other":
        status, reason = "manual_review", "Recognized as outside the supported model list."
    elif not passes:
        status, reason = (
            "manual_review",
            "Insufficient score or similarity to reviewed training photos.",
        )
    elif suggestion in AMBIGUOUS:
        status, reason = (
            "family_only",
            "MTZ-80/82/820 require the front axle and model marking to be checked.",
        )
    elif not exact_allowed(suggestion, score, similarity, head.get("exact_policy", {})):
        status, reason = (
            "manual_review",
            "Class-specific OOF checks do not support an exact answer.",
        )
    else:
        status, reason = "experimental_match", "Passed the experimental classifier checks."
    return {
        "status": status,
        "model": suggestion if status == "experimental_match" else None,
        "family": "mtz_belarus" if status == "family_only" else None,
        "suggestion": suggestion,
        "score": score,
        "similarity": similarity,
        "reason": reason,
        "alternatives": [
            {"model": labels[int(i)], "score": float(probability[i])}
            for i in np.argsort(probability)[::-1]
            if labels[int(i)] != "other"
        ][:3],
    }


def load_head(path, weights):
    head = json.loads(path.read_text(encoding="utf-8"))
    feature_encoder = head.get("feature_encoder", "dinov2")
    if feature_encoder not in {"dinov2", "convnext", "fused"}:
        raise ValueError("Unsupported feature encoder.")
    if feature_encoder == "fused" and head.get("secondary_encoder_sha256") != digest(
        CONVNEXT_WEIGHTS
    ):
        raise ValueError("The secondary encoder weights do not match the classifier.")
    if head.get("encoder_sha256") != digest(weights):
        raise ValueError("The classifier was trained with different encoder weights.")
    if head.get("input") != 224 or head.get("preprocess") != "letterbox, ImageNet normalize":
        raise ValueError("Unsupported image preprocessing.")
    if "exact_policy" not in head:
        raise ValueError("Missing class-specific decision calibration.")
    labels = head["classes"]
    if len(labels) != len(set(labels)) or not CLASSES.issubset(labels):
        raise ValueError("Missing or duplicated supported model classes.")
    return head


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    feature_encoder = json.loads(args.head.read_text(encoding="utf-8")).get(
        "feature_encoder", "dinov2"
    )
    weights = (
        CONVNEXT_WEIGHTS
        if feature_encoder == "convnext"
        else ROOT / "weights/dinov2-small.safetensors"
    )
    head = load_head(args.head, weights)
    torch.set_num_threads(5)
    files = {str(path.resolve()): path for path in args.images}
    primary = "convnext" if feature_encoder == "convnext" else "dinov2"
    vectors = embeddings(encoder(primary, weights), files, pad=True)
    if feature_encoder == "fused":
        secondary = embeddings(encoder("convnext", CONVNEXT_WEIGHTS), files, pad=True)
        vectors = {key: np.concatenate((value, secondary[key])) for key, value in vectors.items()}
    result = {
        "experimental": True,
        "variant": head["variant"],
        "head_sha256": digest(args.head),
        "images": [
            {"path": path, "sha256": digest(files[path]), **decide(vector, head)}
            for path, vector in vectors.items()
        ],
    }
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
