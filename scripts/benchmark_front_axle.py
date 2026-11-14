"""Train an orientation-guided front-axle region branch using manual training views.

Orientation labels describe where the front is in the image, not axle drive type.
The output is a front-region proposal; occluded axles must still be reviewed.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from scripts.benchmark_grouped_recognition import digest, embeddings, encoder
from scripts.benchmark_tractor_parts import BASE, OUT, ROOT, fit, marking_candidates, read_rows
from scripts.predict_tractor_conditions import CONVNEXT_WEIGHTS, decide


def front_region(image, side):
    w, h = image.size
    boxes = {
        "L": (0, 0.45, 0.62, 1),
        "R": (0.38, 0.45, 1, 1),
        "center": (0.12, 0.45, 0.88, 1),
        "rear": (0, 0.45, 1, 1),
    }
    if side not in boxes:
        raise ValueError("Unknown orientation")
    a, b, c, d = boxes[side]
    return image.crop((int(a * w), int(b * h), int(c * w), int(d * h)))


def linear(spec, x):
    logits = np.asarray(spec["coef"]) @ x + np.asarray(spec["intercept"])
    return spec["classes"][int(logits.argmax())]


def packed(model):
    return dict(
        coef=model.coef_.tolist(),
        intercept=model.intercept_.tolist(),
        classes=model.classes_.tolist(),
    )


def main():
    torch.set_num_threads(5)
    rows, meta, head = read_rows()
    annotations = json.loads(
        (ROOT / "reports/tractor_parts_manual_review.json").read_text(encoding="utf-8")
    )
    data = np.load(OUT / "features.npz")
    vec = dict(zip(data["keys"], data["values"], strict=True))
    by_key = {r["key"]: r for r in rows}
    if any(
        r["key"] not in by_key
        or by_key[r["key"]]["sha256"] != r["sha256"]
        or by_key[r["key"]]["role"] != r["role"]
        for r in annotations
    ):
        raise ValueError("Manual annotation source or split mismatch")
    annotated = [r for r in annotations if r["role"] == "train"]
    actual_train = {r["key"]: r for r in rows if r["role"] == "train"}
    if any(
        r["key"] not in actual_train or actual_train[r["key"]]["sha256"] != r["sha256"]
        for r in annotated
    ):
        raise ValueError("Orientation training label outside training split")
    x = np.stack([vec[r["key"] + "/tractor"] for r in annotated])
    y = np.array([r["front_side"] for r in annotated])
    model = LogisticRegression(C=1, max_iter=2000, class_weight="balanced").fit(x, y)
    folds = StratifiedGroupKFold(3, shuffle=True, random_state=83)
    oof = []
    for a, b in folds.split(x, y, [r["group"] for r in annotated]):
        m = LogisticRegression(C=1, max_iter=2000, class_weight="balanced").fit(x[a], y[a])
        oof.extend(
            dict(key=annotated[i]["key"], expected=y[i], prediction=p)
            for i, p in zip(b, m.predict(x[b]), strict=True)
        )
    orientation = {r["key"]: str(model.predict(vec[r["key"] + "/tractor"][None])[0]) for r in rows}
    orientation_test = [
        dict(
            key=r["key"],
            expected=r["front_side"],
            prediction=orientation[r["key"]],
            axle_visibility=r["axle_visibility"],
        )
        for r in annotations
        if r["role"] == "test"
    ]
    files = {}
    for r in rows:
        path = OUT / "views" / r["sha256"] / "front.jpg"
        with Image.open(path.parent / "tractor.jpg") as im:
            front_region(im, orientation[r["key"]]).save(path, quality=95)
        files[r["key"]] = path
    first = embeddings(encoder("dinov2", ROOT / "weights/dinov2-small.safetensors"), files, True)
    second = embeddings(encoder("convnext", CONVNEXT_WEIGHTS), files, True)
    f = {k: np.concatenate((v, second[k])) for k, v in first.items()}
    np.savez_compressed(OUT / "front_features.npz", keys=list(f), values=list(f.values()))
    train = [r for r in rows if r["role"] == "train"]
    test = [r for r in rows if r["role"] == "test"]
    views = {
        name: np.stack([vec[r["key"] + "/" + name] for r in train]) for name in ("tractor", "lower")
    }
    q = {
        name: np.stack([vec[r["key"] + "/" + name] for r in test]) for name in ("tractor", "lower")
    }
    cache = np.load(BASE / "features.npz")
    full = dict(zip(cache["keys"], cache["values"], strict=True))
    views["full"] = np.stack([full[r["key"]] for r in train])
    q["full"] = np.stack([full[r["key"]] for r in test])
    views["front"] = np.stack([f[r["key"]] for r in train])
    q["front"] = np.stack([f[r["key"]] for r in test])
    experiments = {"front": ("front",), "full_tractor_front": ("full", "tractor", "front")}
    comparison = json.loads((OUT / "comparison.json").read_text(encoding="utf-8"))
    records = {r["key"]: r for r in comparison["predictions"]}
    for key, record in records.items():
        if decide(full[key], head)["suggestion"] != record["baseline"]:
            raise ValueError("Frozen baseline does not match the cached features")
    heads = {}
    for name, parts in experiments.items():
        m = fit(train, np.concatenate([views[p] for p in parts], axis=1))
        pred = m.predict(np.concatenate([q[p] for p in parts], axis=1))
        heads[name] = packed(m)
        for r, label in zip(test, pred, strict=True):
            records[r["key"]][name] = str(label)
    for r in records.values():
        r["ocr"]["candidates_before_context"] = r["ocr"]["candidates"]
        candidates = marking_candidates(r["ocr"]["lines"])
        r["ocr"]["candidates"] = candidates
        conflict = bool(candidates and (len(candidates) != 1 or r["baseline"] not in candidates))
        r["baseline_with_ocr_veto_exact"] = bool(r["baseline_exact"] and not conflict)
        agrees = r["baseline"] == r["tractor"]["prediction"] == r["full_tractor_front"]
        no_conflict = not candidates or (len(candidates) == 1 and r["baseline"] in candidates)
        r["agreement_exact"] = bool(r["baseline_exact"] and agrees and no_conflict)
    provenance = dict(
        crop_head_sha256=digest(OUT / "tractor_head.npz"),
        detector_sha256=digest(ROOT / "weights/grounding-dino-tiny/model.safetensors"),
        encoder_sha256=digest(ROOT / "weights/dinov2-small.safetensors"),
        secondary_encoder_sha256=digest(CONVNEXT_WEIGHTS),
        baseline_head_sha256=digest(BASE / "expanded.json"),
        annotation_sha256=digest(ROOT / "reports/tractor_parts_manual_review.json"),
        detector=dict(shortest_edge=320, longest_edge=544, selection="foreground"),
        preprocess="ImageOps.pad224, ImageNet normalize",
        crop_padding=0.06,
        training=[r["key"] for r in train],
        orientation_training=[r["key"] for r in annotated],
        orientation=packed(model),
        heads=heads,
        limitation="Proposed front region, no confirmed axle type; no exact MTZ upgrade",
    )
    (OUT / "front_bundle.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    from scripts.benchmark_tractor_conditions import CLASSES

    for name in experiments:
        known = [r for r in records.values() if r["expected"] in CLASSES]
        comparison["summary"][name] = dict(
            correct=sum(r[name] == r["expected"] for r in known),
            total=len(known),
            by_class={
                label: dict(
                    correct=sum(r[name] == label for r in known if r["expected"] == label),
                    total=sum(r["expected"] == label for r in known),
                )
                for label in sorted(CLASSES)
            },
            outside_rejected=sum(
                r[name] == "other" for r in records.values() if r["expected"] not in CLASSES
            ),
        )
    comparison["orientation"] = dict(train=len(annotated), oof=oof, test=orientation_test)
    comparison["predictions"] = list(records.values())
    (OUT / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(comparison["summary"], indent=2))


if __name__ == "__main__":
    main()
