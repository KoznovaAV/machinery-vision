"""Evaluate frozen encoders on grouped family, MTZ and condition datasets."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image, ImageOps
from safetensors.torch import load_file
from scripts.build_grouped_family_split import digest, group_rows, read_csv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from src.config.classes import MODEL_CLASSES
from torch import nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    convnext_tiny,
    efficientnet_b0,
)
from torchvision.transforms import v2

FAMILIES = list(MODEL_CLASSES)
MTZ = ["mtz_1221", "mtz_82"]
METHODS = {
    "convnext_crop": ("convnext", False),
    "efficientnet_crop": ("efficientnet", False),
    "dinov2_crop": ("dinov2", False),
    "dinov2_pad": ("dinov2", True),
}
IMAGE_SIZE = 224


def encoder(name: str, dino_weights: Path | None) -> nn.Module:
    if name == "convnext":
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        model.classifier[2] = nn.Identity()
    elif name == "efficientnet":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Identity()
    else:
        if dino_weights is None:
            raise ValueError("DINOv2 requires --dino-weights.")
        model = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=False,
            num_classes=0,
            dynamic_img_size=True,
        )
        model.load_state_dict(load_file(dino_weights), strict=True)
    return model.eval()


def embeddings(
    model: nn.Module, files: dict[str, Path | Image.Image], pad: bool
) -> dict[str, np.ndarray]:
    normalize = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    resize = v2.Compose([v2.Resize(IMAGE_SIZE, antialias=True), v2.CenterCrop(IMAGE_SIZE)])
    result = {}
    items = sorted(files.items())
    for start in range(0, len(items), 24):
        batch = items[start : start + 24]
        tensors = []
        for _, path in batch:
            if isinstance(path, Image.Image):
                image = ImageOps.exif_transpose(path).convert("RGB")
            else:
                with Image.open(path) as handle:
                    image = ImageOps.exif_transpose(handle).convert("RGB")
            if pad:
                image = ImageOps.pad(
                    image,
                    (IMAGE_SIZE, IMAGE_SIZE),
                    method=Image.Resampling.BICUBIC,
                    color=(124, 116, 104),
                )
            else:
                image = resize(image)
            tensors.append(normalize(image))

        with torch.inference_mode():
            values = model(torch.stack(tensors)).numpy()
        result.update(zip((key for key, _ in batch), values, strict=True))
    return result


def arrays(rows: list[dict], features: dict[str, np.ndarray], label: str):
    return (
        np.stack([features[row["path"]] for row in rows]),
        np.array([row[label] for row in rows]),
        np.array([row["group"] for row in rows]),
    )


def classification(y, pred, labels):
    return {
        "correct": int(np.sum(y == pred)),
        "n": len(y),
        "macro_f1": f1_score(y, pred, labels=labels, average="macro"),
        "matrix": confusion_matrix(y, pred, labels=labels).tolist(),
    }


def head():
    return LogisticRegression(C=1, class_weight="balanced", max_iter=1500)


def oof(x, y, groups, *, probabilities=False):
    prediction = np.empty(len(y), dtype=float if probabilities else y.dtype)
    covered = np.zeros(len(y), dtype=bool)
    for train, valid in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(x, y, groups):
        if set(groups[train]) & set(groups[valid]) or len(set(y[train])) != len(set(y)):
            raise RuntimeError("Unusable fold.")
        model = head().fit(x[train], y[train])
        prediction[valid] = (
            model.predict_proba(x[valid])[:, 1] if probabilities else model.predict(x[valid])
        )
        covered[valid] = True
    if not np.all(covered):
        raise RuntimeError("Missing out-of-fold predictions.")
    return prediction


def state_rows(inventory: list[dict], review: dict) -> list[dict]:
    rows = []
    for item in inventory:
        path = item["path"]
        clean = item["tree"] == "processed" and item["split"] == "train"
        dirty = item["tree"] == "dirty_clean" and "/train/" in path and "/dirty/real_" in path
        control = item["tree"] in {"real_clean_probe", "real_dirty_val"}
        if clean or dirty or control:
            rows.append(
                {
                    "path": path,
                    "sha256": item["sha256"],
                    "review_id": item["review_id"],
                    "hashes": (int(item["phash"], 16), int(item["phash_flip"], 16)),
                    "state": int(dirty or item["tree"] == "real_dirty_val"),
                    "control": control,
                    "flags": review[item["review_id"]]["flags"].split(";"),
                }
            )
    group_rows(rows, 8)
    return rows


def score_state(train, control, features, output: Path | None = None):
    x, y, groups = arrays(train, features, "state")
    x_control, y_control, _ = arrays(control, features, "state")
    scores = oof(x, y, groups, probabilities=True)
    fpr, tpr, thresholds = roc_curve(y, scores, drop_intermediate=False)
    eligible = np.flatnonzero((fpr <= 0.10) & np.isfinite(thresholds))
    threshold = float(thresholds[max(eligible, key=lambda i: (tpr[i], -fpr[i], thresholds[i]))])
    model = head().fit(x, y)
    pred = model.predict_proba(x_control)[:, 1]

    def matrix(target, score, cutoff):
        return confusion_matrix(target, score >= cutoff, labels=[0, 1]).tolist()

    if output:
        output.write_text(
            json.dumps(
                {
                    "classes": [0, 1],
                    "threshold": threshold,
                    "coef": model.coef_[0].tolist(),
                    "intercept": float(model.intercept_[0]),
                    "weights": "ConvNeXt-Tiny ImageNet-1K V1",
                    "input": IMAGE_SIZE,
                    "preprocess": "resize short edge, center crop, ImageNet normalize",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return {
        "counts_train": dict(Counter(map(int, y))),
        "counts_control": dict(Counter(map(int, y_control))),
        "threshold_oof": threshold,
        "matrix_oof": matrix(y, scores, threshold),
        "matrix_control_0_5": matrix(y_control, pred, 0.5),
        "matrix_control_oof_threshold": matrix(y_control, pred, threshold),
        "auc_oof": roc_auc_score(y, scores),
        "auc_control": roc_auc_score(y_control, pred),
        "control_scores": {
            row["path"]: float(score) for row, score in zip(control, pred, strict=True)
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--audit", type=Path, default=Path("output/dataset_audit"))
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--dino-weights", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, action="append")
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    family_rows = [r for r in read_csv(args.split) if r["split"] != "excluded"]
    mtz_rows = [
        r
        for r in read_csv(args.audit / "mtz_pilot_candidates.csv")
        if r["pilot_status"] == "candidate"
    ]
    for row in mtz_rows:
        row["path"] = row["image"]
        row["group"] = row["duplicate_group"]
    collection = read_csv(args.audit / "collection/manifest.csv")
    for row in collection:
        row["path"] = "collection/" + row["file"]
        row["label"] = next(
            (
                label
                for prefix, label in (("Belarus 82", "mtz_82"), ("Belarus 1221", "mtz_1221"))
                if row["model_on_source"].startswith(prefix)
            ),
            "",
        )
    collection = [row for row in collection if row["label"]]
    if not collection:
        parser.error("The collection manifest must contain at least one supported MTZ model.")
    backing = read_csv(args.audit / "inventory.csv")
    visual = {r["review_id"]: r for r in read_csv(args.audit / "visual_review.csv")}
    state = state_rows(backing, visual)
    control = [r for r in state if r["control"]]
    control_groups = {r["group"] for r in control}
    train = [r for r in state if not r["control"] and r["group"] not in control_groups]
    strict = [
        r
        for r in train
        if not (
            {
                "snow",
                "likely_game_render",
                "dirty_label_review" if r["state"] else "dirty_or_work_scene",
                "catalog_or_render_review",
            }
            & set(r["flags"])
        )
    ]
    files = {r["path"]: args.data / r["path"] for r in family_rows + train + control}
    files.update((r["path"], args.audit / r["path"]) for r in collection)
    inventory_hashes = {r["path"]: r["sha256"] for r in backing}
    inventory_hashes.update((r["path"], r["sha256"]) for r in collection)
    for key, path in files.items():
        if digest(path) != inventory_hashes[key]:
            raise RuntimeError(f"File has changed since the audit: {key}")
    result = {
        "family_labels": FAMILIES,
        "mtz_labels": MTZ,
        "state_train_files_excluded_for_overlap": len(state) - len(control) - len(train),
        "encoders": {},
        "dinov2_sha256": digest(args.dino_weights) if args.dino_weights else None,
    }
    methods = args.method or (
        list(METHODS) if args.dino_weights else ["convnext_crop", "efficientnet_crop"]
    )
    if any(METHODS[key][0] == "dinov2" for key in methods) and not args.dino_weights:
        parser.error("DINOv2 methods require --dino-weights.")
    model = None
    loaded = None
    for key in methods:
        name, pad = METHODS[key]
        start = time.monotonic()
        if loaded != name:
            model = encoder(name, args.dino_weights)
            loaded = name
        print(f"Encoding {len(files)} images with {key}", flush=True)
        feats = embeddings(model, files, pad)
        family_train = [r for r in family_rows if r["split"] == "train"]
        x, y, _ = arrays(family_train, feats, "family")
        family_head = head().fit(x, y)
        metrics = {"family": {}}
        for subset in ("val", "test"):
            x, y, _ = arrays([r for r in family_rows if r["split"] == subset], feats, "family")
            metrics["family"][subset] = classification(y, family_head.predict(x), FAMILIES)
        x, y, groups = arrays(mtz_rows, feats, "historical_model_candidate")
        metrics["mtz_group_cv"] = classification(y, oof(x, y, groups), MTZ)
        model_head = head().fit(x, y)
        if key == "dinov2_pad":
            mtz_candidate = {
                "classes": model_head.classes_.tolist(),
                "coef": model_head.coef_[0].tolist(),
                "intercept": float(model_head.intercept_[0]),
                "input": IMAGE_SIZE,
                "encoder": "dinov2",
                "encoder_sha256": result["dinov2_sha256"],
                "preprocess": "letterbox, ImageNet normalize",
            }
            (args.output / "mtz_candidate.json").write_text(
                json.dumps(mtz_candidate, indent=2) + "\n", encoding="utf-8"
            )
        ext = np.stack([feats[row["path"]] for row in collection])
        metrics["mtz_external"] = [
            {"file": row["file"], "source_label": row["label"], "prediction": pred}
            for row, pred in zip(collection, model_head.predict(ext), strict=True)
        ]
        export = args.output / "state_candidate.json" if name == "convnext" else None
        metrics["state_all_real"] = score_state(train, control, feats)
        metrics["state_screened_real"] = score_state(strict, control, feats, export)
        metrics["embedding_seconds"] = round(time.monotonic() - start, 2)
        result["encoders"][key] = metrics
        (args.output / "grouped_benchmark.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(
            key,
            metrics["family"]["val"]["macro_f1"],
            metrics["family"]["test"]["macro_f1"],
            metrics["mtz_group_cv"]["macro_f1"],
            "seconds",
            metrics["embedding_seconds"],
            flush=True,
        )


if __name__ == "__main__":
    main()
