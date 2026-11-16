"""Compare dirty classifiers with and without synthetic training images."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from hashlib import file_digest
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from torch import nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
from torchvision.transforms import v2

from src.config.classes import MODEL_CLASSES
from src.data.dataset import IMAGE_EXTENSIONS
from src.data.transforms import get_val_transforms
from src.models.loader import load_multi_task_model


def images(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def family(path: Path, root: Path) -> str:
    return next(name for name in path.relative_to(root).parts[:-1] if name in MODEL_CLASSES)


def check_partitions(training: list[Path], evaluation: list[Path]) -> None:
    seen = {sha(p) for p in training}
    if any(sha(p) in seen for p in evaluation):
        raise ValueError(
            "Exact training/evaluation overlap. Review the dataset before comparing models."
        )


def encode(paths: list[Path], batch_size: int, device: str) -> dict[Path, np.ndarray]:
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1).eval().to(device)
    model.classifier[2] = nn.Identity()
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(192, antialias=True),
            v2.CenterCrop(192),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    features = {}
    for start in range(0, len(paths), batch_size):
        batch = paths[start : start + batch_size]
        inputs = []
        for path in batch:
            with Image.open(path) as image:
                inputs.append(transform(ImageOps.exif_transpose(image).convert("RGB")))
        with torch.inference_mode():
            vectors = model(torch.stack(inputs).to(device)).cpu().numpy()
        features.update(zip(batch, vectors, strict=True))
        if start % (10 * batch_size) == 0:
            print(f"Encoded {start + len(batch)}/{len(paths)}", flush=True)
    return features


def metrics(labels: list[int], scores: np.ndarray) -> dict:
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "confusion": {
            f"{threshold:.2f}": confusion_matrix(
                labels, scores >= threshold, labels=[0, 1]
            ).tolist()
            for threshold in (0.50, 0.60)
        },
    }


def reference_scores(
    checkpoint: Path, paths: list[Path], device: str
) -> tuple[np.ndarray, list[str]]:
    model = load_multi_task_model(checkpoint, device=device)
    transform = get_val_transforms()
    scores = []
    predictions = []
    for start in range(0, len(paths), 16):
        batch = []
        for path in paths[start : start + 16]:
            with Image.open(path) as image:
                batch.append(transform(image=np.asarray(image.convert("RGB")))["image"])
        with torch.inference_mode():
            families, states = model(torch.stack(batch).to(device))
        predictions.extend(MODEL_CLASSES[idx] for idx in families.argmax(1).tolist())
        scores.extend(states.softmax(1)[:, 1].tolist())
    return np.asarray(scores), predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--threads", default=6, type=int)
    parser.add_argument("--batch-size", default=24, type=int)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    clean_train = images(args.data / "processed/train")
    dirty_train = [p for p in images(args.data / "dirty_clean/train") if p.parent.name == "dirty"]
    real = [p for p in dirty_train if p.name.startswith("real_")]
    synthetic = [p for p in dirty_train if re.search(r"_dirty\d+$", p.stem)]
    clean_eval = images(args.data / "real_clean_probe")
    dirty_eval = images(args.data / "real_dirty_val")
    family_eval = images(args.data / "processed/test")
    if not all((clean_train, real, synthetic, clean_eval, dirty_eval, family_eval)):
        parser.error("Missing expected train, test or real control images.")
    evaluation = clean_eval + dirty_eval
    check_partitions(clean_train + real + synthetic, evaluation)
    features = encode(
        sorted(set(clean_train + real + synthetic + evaluation)), args.batch_size, args.device
    )
    eval_features = np.stack([features[path] for path in evaluation])
    labels = [0] * len(clean_eval) + [1] * len(dirty_eval)

    variants = [
        ("real_only", real),
        ("real_plus_all_synthetic", real + synthetic),
        ("synthetic_only", synthetic),
    ]
    # Hold the real examples fixed while changing the sample of synthetic positives.
    for seed in (21, 42, 63):
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(synthetic))
        for count in (len(real) // 4, len(real), 3 * len(real)):
            subset = [synthetic[i] for i in order[:count]]
            if subset and len(subset) < len(synthetic):
                variants.append((f"real_plus_{len(subset)}_synthetic_seed_{seed}", real + subset))

    output = {
        "features": "ImageNet ConvNeXt-Tiny, frozen, 192 px center crop",
        "real_positive_train_by_family": dict(Counter(family(path, args.data) for path in real)),
        "evaluation": {"clean": len(clean_eval), "dirty": len(dirty_eval)},
        "confusion_order": [["true_clean", "false_dirty"], ["missed_dirty", "true_dirty"]],
        "variants": {},
    }
    for name, positives in variants:
        training = clean_train + positives
        targets = [0] * len(clean_train) + [1] * len(positives)
        classifier = LogisticRegression(C=1, class_weight="balanced", max_iter=1500)
        classifier.fit(np.stack([features[p] for p in training]), targets)
        scores = classifier.predict_proba(eval_features)[:, 1]
        output["variants"][name] = {
            "train_clean": len(clean_train),
            "train_real_dirty": sum(p.name.startswith("real_") for p in positives),
            "train_synthetic_dirty": sum(not p.name.startswith("real_") for p in positives),
            **metrics(labels, scores),
        }

    if args.checkpoint:
        scores, predictions = reference_scores(
            args.checkpoint, evaluation + family_eval, args.device
        )
        true = [family(p, args.data) for p in family_eval]
        predicted = predictions[len(evaluation) :]
        output["reference"] = {
            "checkpoint_sha256": sha(args.checkpoint),
            "state": metrics(labels, scores[: len(evaluation)]),
            "family_order": list(MODEL_CLASSES),
            "family_accuracy": float(np.mean(np.asarray(true) == predicted)),
            "family_macro_f1": float(
                f1_score(true, predicted, labels=MODEL_CLASSES, average="macro")
            ),
            "family_confusion": confusion_matrix(true, predicted, labels=MODEL_CLASSES).tolist(),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
