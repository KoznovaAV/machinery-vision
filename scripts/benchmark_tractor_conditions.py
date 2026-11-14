"""Compare tractor recognition with added photos, augmentation and an option to abstain."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from scripts.benchmark_grouped_recognition import digest, embeddings, encoder, read_csv
from scripts.benchmark_mtz_models import ROOT, add_historical, load_rows
from scripts.train_open_set_filter import unit

BASE = ROOT / "output/weather_expansion"
CLASSES = {"mtz_80", "mtz_82", "mtz_820", "mtz_1221", "nh_t5000", "claas_axion_800"}
TEST = {"test", "test_wave", "external_test", "weather_test", "snow", "outside", "outside_wave"}


def augment(path):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    image.thumbnail((500, 500))
    gray = ImageOps.grayscale(image).convert("RGB")
    low_light = ImageEnhance.Brightness(ImageEnhance.Contrast(image).enhance(0.72)).enhance(0.73)
    return ImageOps.mirror(image), gray, low_light


def train(rows, vectors, augmented=False):
    counts = Counter(r["group"] for r in rows)
    weights = np.array([1 / counts[r["group"]] for r in rows])
    targets = np.array([r["model"] for r in rows])
    if not CLASSES.issubset(set(targets)):
        raise ValueError("Training is missing a supported class.")
    for label in sorted(set(targets)):
        selected = targets == label
        if not selected.any():
            raise ValueError(f"No examples for {label}")
        weights[selected] /= weights[selected].sum()
    weights *= len(rows) / weights.sum()
    x = []
    y = []
    w = []
    for r, target, weight in zip(rows, targets, weights, strict=True):
        keys = [r["key"]]
        if augmented:
            keys += [f"{r['key']}@{j}" for j in range(3)]
        for key in keys:
            x.append(vectors[key])
            y.append(target)
            w.append(weight / len(keys))
    return LogisticRegression(C=1, max_iter=2500).fit(np.stack(x), y, sample_weight=w)


def confidence(model, train_rows, vectors, query):
    x = np.stack([vectors[r["key"]] for r in query])
    prob = model.predict_proba(x)
    best = prob.argmax(axis=1)
    labels = model.classes_[best]
    evidence = unit([vectors[r["key"]] for r in train_rows])
    similarity = unit(x) @ evidence.T
    for i, label in enumerate(labels):
        subset = np.array([r["model"] == label for r in train_rows])
        similarity[i, ~subset] = -1
    return labels, prob.max(axis=1), similarity.max(axis=1)


def choose_cutoff(known, unknown, already_rejected=None):
    # Keep a usable fraction of correct OOF decisions while rejecting calibration negatives.
    if already_rejected is None:
        already_rejected = np.zeros(len(unknown), dtype=bool)
    if not len(known) or not len(unknown):
        raise ValueError("Calibration needs correct known and negative examples.")
    choices = [
        (p, cos)
        for p in np.quantile(known[:, 0], np.linspace(0, 0.5, 11))
        for cos in np.quantile(known[:, 1], np.linspace(0, 0.5, 11))
    ]
    valid = []
    for p, cos in choices:
        covered = float(np.mean((known[:, 0] >= p) & (known[:, 1] >= cos)))
        if covered >= 0.85:
            rejected = float(
                np.mean(already_rejected | (unknown[:, 0] < p) | (unknown[:, 1] < cos))
            )
            valid.append((rejected, covered, float(p), float(cos)))
    return max(valid)


def calibration(rows, vectors):
    folds = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=83)
    labels = [r["model"] for r in rows]
    groups = [r["group"] for r in rows]
    scores = []
    records = []
    for train_idx, validation_idx in folds.split(rows, labels, groups):
        subtrain = [rows[i] for i in train_idx]
        validation = [rows[i] for i in validation_idx]
        model = train(subtrain, vectors, augmented=True)
        pred, probability, similarity = confidence(model, subtrain, vectors, validation)
        scores.extend(
            (p, cos)
            for r, label, p, cos in zip(validation, pred, probability, similarity, strict=True)
            if label == r["model"] and label != "other"
        )
        records.extend(
            dict(
                key=r["key"],
                group=r["group"],
                expected=r["model"],
                prediction=label,
                score=float(p),
                similarity=float(cos),
            )
            for r, label, p, cos in zip(validation, pred, probability, similarity, strict=True)
        )
    return np.asarray(scores), records


def choose_class_cutoffs(records, target_precision=0.95):
    # An empirical OOF precision target, not a deployment accuracy guarantee.
    policy = {}
    for label in sorted(CLASSES):
        selected = [r for r in records if r["prediction"] == label]
        disabled = dict(enabled=False, reason="Too few reliable independent OOF decisions")
        policy[label] = disabled
        if not selected:
            continue
        scores = np.asarray([r["score"] for r in selected])
        similarities = np.asarray([r["similarity"] for r in selected])
        correct = np.asarray([r["expected"] == label for r in selected])
        counts = Counter(r["group"] for r in selected)
        weights = np.asarray([1 / counts[r["group"]] for r in selected])
        options = []
        for p in np.unique(np.quantile(scores, np.linspace(0, 1, 21))):
            for cos in np.unique(np.quantile(similarities, np.linspace(0, 1, 21))):
                keep = (scores >= p) & (similarities >= cos)
                groups = {r["group"] for i, r in enumerate(selected) if keep[i]}
                if keep.sum() < 8 or len(groups) < 3:
                    continue
                precision = float(weights[keep & correct].sum() / weights[keep].sum())
                if precision >= target_precision:
                    options.append(
                        (
                            float(weights[keep & correct].sum()),
                            precision,
                            float(p),
                            float(cos),
                            int(keep.sum()),
                            len(groups),
                        )
                    )
        if options:
            _, precision, p, cos, n, n_groups = max(options)
            policy[label] = dict(
                enabled=True,
                min_probability=p,
                min_similarity=cos,
                oof_precision=precision,
                oof_accepted=n,
                oof_groups=n_groups,
                target_precision=target_precision,
            )
    return policy


def exact_allowed(label, score, similarity, policy):
    spec = policy.get(label, {})
    return bool(
        spec.get("enabled")
        and score >= spec["min_probability"]
        and similarity >= spec["min_similarity"]
    )


def validate_training(train_rows, control_rows):
    if {r["group"] for r in train_rows} & {r["group"] for r in control_rows}:
        raise ValueError("A machine/source series crosses training and evaluation.")
    if {r["sha256"] for r in train_rows} & {r["sha256"] for r in control_rows}:
        raise ValueError("A training image also appears in evaluation.")
    if len({r["sha256"] for r in train_rows}) != len(train_rows):
        raise ValueError("Duplicate training images.")


def reference_digests(prior_rows, current_rows):
    current = {r["sha256"]: (r["group"], r["model"], r["split"]) for r in current_rows}
    for row in prior_rows:
        if current.get(row["sha256"]) != (row["group"], row["model"], row["split"]):
            raise ValueError("A prior image, label or partition was changed.")
    return {r["sha256"] for r in prior_rows}


def reusable_vectors(path, encoder_hash, source_hashes, keys, dimensions):
    """Reuse original and augmented embeddings only for unchanged source pixels."""
    with np.load(path, allow_pickle=False) as cache:
        signature = str(cache["signature"])
        marker = "|augment-v1-pad224|"
        if not signature.startswith(encoder_hash + marker):
            return {}
        if cache["values"].shape[1] != dimensions:
            return {}
        previous = {}
        for item in signature.split(marker, 1)[1].split("|"):
            key, sha = item.rsplit(":", 1)
            previous[key] = sha
        available = dict(zip(cache["keys"].tolist(), cache["values"], strict=True))
    reused = {}
    for key in keys:
        original = key if key in source_hashes else key.rsplit("@", 1)[0]
        if (
            key in available
            and original in source_hashes
            and source_hashes[original] == previous.get(original)
        ):
            reused[key] = available[key]
    return reused


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extra", type=Path)
    parser.add_argument("--encoder", choices=("dinov2", "convnext", "fused"), default="dinov2")
    parser.add_argument("--output", type=Path, default=BASE)
    parser.add_argument("--prior-manifest", type=Path)
    parser.add_argument("--warm-cache", type=Path)
    parser.add_argument("--fusion-cache-root", type=Path, default=BASE)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifests = [
        ROOT / "output/mtz_model_pilot/reviewed.csv",
        ROOT / "output/model_wave2/reviewed.csv",
    ]
    if args.extra:
        manifests.append(args.extra)
    rows = load_rows(manifests)
    add_historical(rows)
    prior = reference_digests(read_csv(args.prior_manifest), rows) if args.prior_manifest else set()
    new_digests = (
        {r["sha256"] for r in read_csv(args.extra)} - prior
        if args.prior_manifest and args.extra
        else set()
    )

    conditions = {}
    if args.extra:
        conditions = {
            r["sha256"]: (r["condition"], r["view"], r["source"]) for r in read_csv(args.extra)
        }
    old_train = [r for r in rows if r["split"] in {"train", "historical", "train_wave"}]
    expanded_train = old_train + [r for r in rows if r["split"] == "external_train"]
    control = [r for r in rows if r["split"] in TEST]
    if {r["sha256"] for r in expanded_train} & {r["sha256"] for r in control}:
        raise ValueError("Training and evaluation share an image.")

    negatives = []
    known_digests = {r["sha256"] for r in rows}
    for r in read_csv(ROOT / "output/open_set/reviewed_candidates.csv"):
        if r["split"] != "val" or r["category"].casefold() not in {"zetor", "dt-75", "t-150k"}:
            continue
        path = ROOT / "output/open_set" / r["file"]
        if r["sha256"] in known_digests or digest(path) != r["sha256"]:
            raise ValueError(f"Calibration photo overlaps training or has changed: {path}")
        negatives.append(
            dict(key="reject-cal/" + r["id"], path=path, group=r["group"], sha256=r["sha256"])
        )
    for r in negatives:
        r["sha256"] = digest(r["path"])
    negative_train = []
    for r in read_csv(ROOT / "output/open_set/reviewed_candidates.csv"):
        if r["split"] != "train" or r["category"].casefold() not in {"dt-75", "t-150k"}:
            continue
        path = ROOT / "output/open_set" / r["file"]
        if digest(path) != r["sha256"] or r["sha256"] in known_digests:
            raise ValueError("Changed or overlapping negative training image.")
        negative_train.append(
            dict(
                key="negative-train/" + r["id"],
                path=path,
                group=r["group"],
                model="other",
                split="negative_train",
                sha256=r["sha256"],
            )
        )
    for r in rows:
        if r["split"] == "negative_train":
            negative_train.append({**r, "model": "other"})
    validate_training(expanded_train + negative_train, control + negatives)
    if not negatives:
        raise ValueError("No negative calibration controls.")

    files = {r["key"]: r["path"] for r in rows + negatives + negative_train}
    for r in expanded_train + negative_train:
        for j, image in enumerate(augment(r["path"])):
            files[f"{r['key']}@{j}"] = image
    torch.set_num_threads(5)
    weights = (
        ROOT / "weights/dinov2-small.safetensors"
        if args.encoder != "convnext"
        else Path.home() / ".cache/torch/hub/checkpoints/convnext_tiny-983f1562.pth"
    )
    print(f"Encoding {len(files)} originals and augmented views", flush=True)
    signature = (
        digest(weights)
        + "|augment-v1-pad224|"
        + "|".join(
            f"{r['key']}:{r.get('sha256', digest(r['path']))}"
            for r in sorted(rows + negatives + negative_train, key=lambda row: row["key"])
        )
    )
    cache_path = args.output / "features.npz"
    vectors = None
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cache:
            dimensions = {"dinov2": 384, "convnext": 768, "fused": 1152}
            secondary_matches = args.encoder != "fused" or (
                "secondary_encoder_sha256" in cache
                and str(cache["secondary_encoder_sha256"])
                == digest(Path.home() / ".cache/torch/hub/checkpoints/convnext_tiny-983f1562.pth")
            )
            if (
                str(cache["signature"]) == signature
                and secondary_matches
                and cache["values"].shape == (len(files), dimensions[args.encoder])
                and set(cache["keys"].tolist()) == set(files)
            ):
                vectors = dict(zip(cache["keys"].tolist(), cache["values"], strict=True))
    if vectors is None:
        if args.encoder == "fused":
            cache_roots = (args.fusion_cache_root / "refined", args.fusion_cache_root / "convnext")
            feature_sets = []
            for directory in cache_roots:
                with np.load(directory / "features.npz", allow_pickle=False) as cached:
                    expected_weight = (
                        ROOT / "weights/dinov2-small.safetensors"
                        if directory.name == "refined"
                        else Path.home() / ".cache/torch/hub/checkpoints/convnext_tiny-983f1562.pth"
                    )
                    if str(cached["signature"]) != digest(expected_weight) + signature[64:]:
                        raise ValueError("Fusion cache does not match current inputs.")
                    feature_sets.append(
                        dict(zip(cached["keys"].tolist(), cached["values"], strict=True))
                    )
            vectors = {
                key: np.concatenate([features[key] for features in feature_sets]) for key in files
            }
        else:
            source_hashes = {r["key"]: r["sha256"] for r in rows + negatives + negative_train}
            vectors = (
                reusable_vectors(
                    args.warm_cache,
                    digest(weights),
                    source_hashes,
                    files,
                    {"dinov2": 384, "convnext": 768}[args.encoder],
                )
                if args.warm_cache
                else {}
            )
            missing = {key: file for key, file in files.items() if key not in vectors}
            print(f"Reused {len(vectors)} embeddings; encoding {len(missing)}", flush=True)
            if missing:
                vectors.update(embeddings(encoder(args.encoder, weights), missing, pad=True))
        keys = sorted(vectors)
        np.savez_compressed(
            cache_path,
            signature=signature,
            feature_encoder=args.encoder,
            secondary_encoder_sha256=(
                digest(Path.home() / ".cache/torch/hub/checkpoints/convnext_tiny-983f1562.pth")
                if args.encoder == "fused"
                else ""
            ),
            keys=np.asarray(keys),
            values=np.stack([vectors[k] for k in keys]),
        )

    all_rows = {r["key"]: r for r in rows + negative_train}
    for r in negatives:
        all_rows[r["key"]] = {**r, "model": "other", "split": "calibration"}
    partition_rows = [
        {
            **{k: r[k] for k in ("key", "group", "model", "split", "sha256")},
            "path": str(r["path"].relative_to(ROOT)),
        }
        for r in all_rows.values()
    ]
    payload = {
        "partitions": partition_rows,
        "feature_encoder": args.encoder,
        "encoder_sha256": digest(weights),
        "script_sha256": digest(Path(__file__)),
        "manifest_sha256": {
            str(m): digest(m)
            for m in manifests
            + ([args.prior_manifest] if args.prior_manifest else [])
            + [
                ROOT / "output/mtz_expansion/reviewed_manifest.csv",
                ROOT / "output/dataset_audit/mtz_pilot_candidates.csv",
                ROOT / "output/photo_pilot/downloaded_metadata.json",
                ROOT / "output/open_set/reviewed_candidates.csv",
            ]
        },
        "changes": {"new_train": len(expanded_train) - len(old_train), "control": len(control)},
        "runs": {},
    }
    runs = (
        ("baseline", old_train, False),
        ("added_photos", expanded_train, False),
        ("augmented", expanded_train, True),
        ("hard_negatives", expanded_train + negative_train, True),
        (
            "curated_negatives",
            [r for r in expanded_train if r["split"] != "historical"] + negative_train,
            True,
        ),
    )
    if args.prior_manifest:
        reference_positive = [
            r
            for r in expanded_train
            if r["split"] in {"train", "historical", "train_wave"} or r["sha256"] in prior
        ]
        reference_negative = [
            r
            for r in negative_train
            if r["key"].startswith("negative-train/") or r["sha256"] in prior
        ]
        runs = (
            ("previous", reference_positive + reference_negative, True),
            (
                "previous_without_history",
                [r for r in reference_positive if r["split"] != "historical"] + reference_negative,
                True,
            ),
            ("expanded", expanded_train + negative_train, True),
            (
                "expanded_without_history",
                [r for r in expanded_train if r["split"] != "historical"] + negative_train,
                True,
            ),
        )
        payload["changes"]["added_since_previous"] = len(expanded_train + negative_train) - len(
            reference_positive + reference_negative
        )
    payload["training"] = {}
    for name, train_rows, augmented in runs:
        payload["training"][name] = {
            "images": dict(Counter(r["model"] for r in train_rows)),
            "groups": {
                label: len({r["group"] for r in train_rows if r["model"] == label})
                for label in sorted({r["model"] for r in train_rows})
            },
        }
        model = train(train_rows, vectors, augmented)
        labels, probabilities, similarities = confidence(model, train_rows, vectors, control)
        cutoff = None
        if augmented:
            val_known, oof_records = calibration(train_rows, vectors)
            negative_labels, p_negative, cos_negative = confidence(
                model, train_rows, vectors, negatives
            )
            cutoff = choose_cutoff(
                val_known, np.column_stack((p_negative, cos_negative)), negative_labels == "other"
            )
            rejected, kept, p, cos = cutoff
            payload["calibration"] = {
                "min_probability": p,
                "min_similarity": cos,
                "known_oof_correct": len(val_known),
                "known_kept": kept,
                "negative_count": len(negatives),
                "negative_rejected": rejected,
            }
            oof_records.extend(
                dict(
                    key=r["key"],
                    group=r["group"],
                    expected="other",
                    prediction=label,
                    score=float(p),
                    similarity=float(cos),
                )
                for r, label, p, cos in zip(
                    negatives, negative_labels, p_negative, cos_negative, strict=True
                )
            )
            exact_policy = choose_class_cutoffs(oof_records)
            payload.setdefault("oof_predictions", {})[name] = oof_records
            payload.setdefault("exact_policies", {})[name] = exact_policy
            head = dict(
                classes=model.classes_.tolist(),
                coef=model.coef_.tolist(),
                intercept=model.intercept_.tolist(),
                encoder_sha256=payload["encoder_sha256"],
                feature_encoder=args.encoder,
                secondary_encoder_sha256=(
                    digest(Path.home() / ".cache/torch/hub/checkpoints/convnext_tiny-983f1562.pth")
                    if args.encoder == "fused"
                    else None
                ),
                input=224,
                preprocess="letterbox, ImageNet normalize",
                reject=payload["calibration"],
                exact_policy=exact_policy,
                known_classes=sorted(CLASSES),
                variant=name,
                training=[
                    {k: r[k] for k in ("key", "sha256", "group", "model", "split")}
                    for r in train_rows
                ],
                train_reference=unit([vectors[r["key"]] for r in train_rows]).tolist(),
                train_reference_classes=[r["model"] for r in train_rows],
            )
            (args.output / f"{name}.json").write_text(json.dumps(head) + "\n", encoding="utf-8")
            payload.setdefault("calibrations", {})[name] = payload["calibration"]
        result = []
        for row, label, p, cos in zip(control, labels, probabilities, similarities, strict=True):
            condition, view, source = conditions.get(row["sha256"], ("unknown", "unknown", ""))
            if row["split"] == "snow" or row["group"] in {"tractorbg-86967", "tractorbg-40262"}:
                condition = "snow"
            accepted = label != "other" and (
                cutoff is None or (p >= cutoff[2] and cos >= cutoff[3])
            )
            precise = bool(
                accepted
                and cutoff is not None
                and label not in {"mtz_80", "mtz_82", "mtz_820"}
                and exact_allowed(label, float(p), float(cos), exact_policy)
            )
            status = (
                "experimental_match"
                if precise
                else (
                    "family_only"
                    if accepted and label in {"mtz_80", "mtz_82", "mtz_820"}
                    else "manual_review"
                )
            )
            result.append(
                dict(
                    key=row["key"],
                    cohort="new" if row["sha256"] in new_digests else "legacy",
                    group=row["group"],
                    split=row["split"],
                    expected=row["model"],
                    prediction=label,
                    score=float(p),
                    similarity=float(cos),
                    accepted=bool(accepted),
                    exact_accepted=precise,
                    status=status,
                    condition=condition,
                    view=view,
                    source=source,
                )
            )
        payload["runs"][name] = result
        print(
            name,
            Counter((r["split"], r["expected"] == r["prediction"], r["accepted"]) for r in result),
            flush=True,
        )
    (args.output / "metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
