"""Experimental tractor crops, lower-body branch and independent OCR evidence.

The lower-body window is a proposal, not an annotated front axle detector.
No OCR or branch prediction is allowed to grant an exact MTZ identity.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression

from scripts.benchmark_grouped_recognition import digest, embeddings, encoder
from scripts.benchmark_tractor_conditions import CLASSES
from scripts.benchmark_tractor_localization import crop, load_detector, tractor_box
from scripts.predict_tractor_conditions import CONVNEXT_WEIGHTS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/tractor_parts"
BASE = ROOT / "output/targeted_weather2/fused"
FOCUS = {"mtz_80", "mtz_82", "mtz_820", "nh_t5000"}


def marking_labels(text):
    """Conservative token parsing: 820 is not 82, and 80.1 is not 80."""
    text = text.upper().replace("Т", "T").replace("О", "O")
    result = set()
    for token in re.findall(r"(?<![A-Z0-9])(?:T\s*)?\d{2,4}(?:[.,]\d+)?(?![A-Z0-9])", text):
        token = token.replace(" ", "").replace(",", ".")
        if re.fullmatch(r"T50[3-7]0", token):
            result.add("nh_t5000")
        elif token in {"80", "80.1"}:
            result.add("mtz_80")
        elif token in {"82", "82.1"}:
            result.add("mtz_82")
        elif token in {"820", "820.1"}:
            result.add("mtz_820")
    if re.search(r"\b(?:TD\s*\d+[A-Z]*|T[4-9]\s*\.\s*\d+[A-Z]*|ARION)\b", text):
        result.add("outside_supported")
    return sorted(result)


def lower_proposal(image):
    # Includes both ends: a fixed window cannot know which side is the front.
    return image.crop((0, int(image.height * 0.48), image.width, image.height))


def read_rows():
    meta = json.loads((BASE / "metrics.json").read_text(encoding="utf-8"))
    head = json.loads((BASE / "expanded.json").read_text(encoding="utf-8"))
    train_keys = {r["key"] for r in head["training"]}
    test_keys = {r["key"] for r in meta["runs"]["expanded"]}
    rows = [r for r in meta["partitions"] if r["key"] in train_keys | test_keys]
    for r in rows:
        if r["key"] in train_keys:
            r["role"] = "train"
        else:
            r["role"] = "test"
        if digest(ROOT / r["path"]) != r["sha256"]:
            raise ValueError("Changed source: " + r["path"])
    if len(rows) != len(train_keys) + len(test_keys):
        raise ValueError("Missing or repeated source rows")
    train = [r for r in rows if r["role"] == "train"]
    test = [r for r in rows if r["role"] == "test"]
    if {r["group"] for r in train} & {r["group"] for r in test}:
        raise ValueError("Source group leakage")
    if {r["sha256"] for r in train} & {r["sha256"] for r in test}:
        raise ValueError("Image leakage")
    return rows, meta, head


def prepare(rows):
    # Load legacy detector dependencies only after importing the project's packages.
    import os

    deps = Path(os.environ["TEMP"]) / "machinery-detector-deps"
    sys.path.insert(0, str(deps))
    detector = load_detector(ROOT / "weights/grounding-dino-tiny")
    for n, r in enumerate(rows):
        folder = OUT / "views" / r["sha256"]
        record_path = folder / "regions.json"
        if record_path.exists():
            saved = json.loads(record_path.read_text())
            if saved["sha256"] != r["sha256"]:
                raise ValueError("Stale detection")
            if saved.get("detector_size") == [320, 544]:
                continue
        folder.mkdir(parents=True, exist_ok=True)
        with Image.open(ROOT / r["path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        det = tractor_box(image, detector, shortest_edge=320, longest_edge=544)
        selected = det.get("foreground")
        tractor = crop(image, selected["box"]) if selected else image
        tractor.save(folder / "tractor.jpg", quality=95)
        lower_proposal(tractor).save(folder / "lower.jpg", quality=95)
        record = dict(
            detector_size=[320, 544],
            sha256=r["sha256"],
            key=r["key"],
            detections=det,
            selected="foreground" if selected else "fallback_full",
            image_size=list(image.size),
            lower_region="bottom 52% of tractor crop; unverified axle proposal",
        )
        record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f'detect {n+1}/{len(rows)} {r["key"]}', flush=True)


def extract(rows):
    cache = OUT / "features.npz"
    signature = digest(Path(__file__)) + "|" + "|".join(r["sha256"] for r in rows)
    if cache.exists():
        data = np.load(cache)
        if str(data["signature"]) == signature:
            return dict(zip(data["keys"], data["values"], strict=True))
    files = {
        r["key"] + "/" + view: OUT / "views" / r["sha256"] / (view + ".jpg")
        for r in rows
        for view in ("tractor", "lower")
    }
    first = embeddings(encoder("dinov2", ROOT / "weights/dinov2-small.safetensors"), files, True)
    second = embeddings(encoder("convnext", CONVNEXT_WEIGHTS), files, True)
    vectors = {k: np.concatenate((v, second[k])) for k, v in first.items()}
    np.savez_compressed(
        cache, signature=signature, keys=list(vectors), values=list(vectors.values())
    )
    return vectors


def ocr(rows):
    import cv2
    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1)
    folder = OUT / "ocr"
    folder.mkdir(exist_ok=True)
    results = {}
    for i, r in enumerate(rows):
        path = folder / (r["sha256"] + ".json")
        if path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            cached["candidates"] = marking_candidates(cached["lines"])
            results[r["key"]] = cached
            continue
        # Only the detected machine, at native resolution. No page text or titles.
        image = Image.open(OUT / "views" / r["sha256"] / "tractor.jpg").convert("RGB")
        rgb = np.asarray(image)
        found, _ = engine(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        lines = [
            dict(
                box=np.asarray(box).tolist(),
                text=text,
                confidence=float(score),
                candidates=marking_labels(text),
            )
            for box, text, score in (found or [])
        ]
        record = dict(
            lines=lines,
            candidates=marking_candidates(lines),
            warning="OCR candidates require visual confirmation; unrelated text possible",
        )
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        results[r["key"]] = record
        print(f'ocr {i+1}/{len(rows)} {r["key"]}', flush=True)
    return results


def fit(rows, features):
    counts = Counter(r["group"] for r in rows)
    y = np.array([r["model"] for r in rows])
    weights = np.array([1 / counts[r["group"]] for r in rows])
    for label in set(y):
        weights[y == label] /= weights[y == label].sum()
    weights *= len(rows) / weights.sum()
    return LogisticRegression(C=1, max_iter=2500).fit(features, y, sample_weight=weights)


def evaluate(rows, meta, head, vectors, readings):
    cache = np.load(BASE / "features.npz")
    full = dict(zip(cache["keys"], cache["values"], strict=True))
    train = [r for r in rows if r["role"] == "train"]
    test = [r for r in rows if r["role"] == "test"]
    x = {
        view: np.stack([vectors[r["key"] + "/" + view] for r in train])
        for view in ("tractor", "lower")
    }
    x["full"] = np.stack([full[r["key"]] for r in train])
    q = {
        view: np.stack([vectors[r["key"] + "/" + view] for r in test])
        for view in ("tractor", "lower")
    }
    q["full"] = np.stack([full[r["key"]] for r in test])
    # Same non-augmented learner for all three ablations; frozen deployed pilot separate.
    experiments = {view: (x[view], q[view]) for view in x}
    experiments["tractor_lower"] = (
        np.concatenate((x["tractor"], x["lower"]), axis=1),
        np.concatenate((q["tractor"], q["lower"]), axis=1),
    )
    experiments["full_tractor_lower"] = (
        np.concatenate(tuple(x.values()), axis=1),
        np.concatenate(tuple(q.values()), axis=1),
    )
    previous = {r["key"]: r for r in meta["runs"]["expanded"]}
    results = []
    for i, r in enumerate(test):
        old = previous[r["key"]]
        results.append(
            dict(
                key=r["key"],
                expected=r["model"],
                group=r["group"],
                baseline=old["prediction"],
                baseline_status=old["status"],
                baseline_exact=old["exact_accepted"],
                ocr=readings[r["key"]],
            )
        )
    for name, (a, b) in experiments.items():
        model = fit(train, a)
        prediction = model.predict(b)
        probability = model.predict_proba(b).max(axis=1)
        for record, label, score in zip(results, prediction, probability, strict=True):
            record[name] = dict(prediction=str(label), probability=float(score))
        np.savez_compressed(
            OUT / (name + "_head.npz"),
            coef=model.coef_,
            intercept=model.intercept_,
            classes=model.classes_,
        )
    for record in results:
        candidates = record["ocr"]["candidates"]
        raw = record["full_tractor_lower"]["prediction"]
        # Evidence may veto a claim. Never upgrade OCR into an exact identity.
        conflict = bool(candidates and (len(candidates) != 1 or raw not in candidates))
        baseline_conflict = bool(
            candidates and (len(candidates) != 1 or record["baseline"] not in candidates)
        )
        record["baseline_with_ocr_veto_exact"] = record["baseline_exact"] and not baseline_conflict
        record["parts_review"] = dict(
            suggestion=raw,
            marking_conflict=conflict,
            axle_status="unverified_lower_body_proposal",
            status="manual_review" if conflict or raw in FOCUS else "uncalibrated_suggestion",
            exact_model=None,
        )
    summary = {}
    for name in ["baseline", *experiments]:

        def get(r):
            return r[name] if name == "baseline" else r[name]["prediction"]

        known = [r for r in results if r["expected"] in CLASSES]
        summary[name] = dict(
            correct=sum(get(r) == r["expected"] for r in known),
            total=len(known),
            by_class={
                label: dict(
                    correct=sum(get(r) == label for r in known if r["expected"] == label),
                    total=sum(r["expected"] == label for r in known),
                )
                for label in sorted(CLASSES)
            },
            outside_rejected=sum(
                get(r) == "other" for r in results if r["expected"] not in CLASSES
            ),
        )
    output = dict(
        experimental=True,
        baseline_head_sha256=digest(BASE / "expanded.json"),
        script_sha256=digest(Path(__file__)),
        train=len(train),
        test=len(test),
        split_policy="unchanged round2 groups, no controls in training",
        summary=summary,
        predictions=results,
    )
    (OUT / "comparison.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["detect", "evaluate", "all"], default="all")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(5)
    rows, meta, head = read_rows()
    if args.stage in {"all", "detect"}:
        prepare(rows)
    if args.stage in {"all", "evaluate"}:
        evaluate(rows, meta, head, extract(rows), ocr([r for r in rows if r["role"] == "test"]))


def marking_candidates(lines):
    """Prefer explicit model codes over isolated numbers on implements or plates.

    This is a review/veto hint, never permission to identify a tractor exactly.
    """
    reliable = [line for line in lines if line["confidence"] >= 0.9]
    explicit = []
    for line in reliable:
        text = line["text"].upper().replace("Т", "T")
        if re.search(r"T\s*50[3-7]0|ARION|\bTD\s*\d|(?:MT3|MTZ|МТЗ|BELARUS)\s*[- ]?\s*\d", text):
            explicit.extend(line["candidates"])
    if explicit:
        return sorted(set(explicit))
    return sorted(
        {
            candidate
            for line in reliable
            if not re.search(r"\d\s*[:;]", line["text"])
            for candidate in line["candidates"]
        }
    )


if __name__ == "__main__":
    main()
