"""Source-group weighting, OOF calibration and candidate training."""

from collections import Counter, defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from src.config.machinery_taxonomy import MTZ_SPECIALIST


def serialize(model):
    return dict(
        classes=model.classes_.tolist(),
        coef=model.coef_.tolist(),
        intercept=model.intercept_.tolist(),
    )


def specialist_rows(rows):
    return [dict(r) for r in rows if r["model"] in MTZ_SPECIALIST]


def calibrate(records, classes, min_groups=10, target_precision=0.95):
    by = defaultdict(list)
    for r in records:
        by[r["group"]].append(r)
    grouped = []
    for group, rr in by.items():
        truth = {r["expected"] for r in rr}
        if len(truth) != 1:
            continue
        scores = np.mean([r["probabilities"] for r in rr], axis=0)
        idx = int(scores.argmax())
        grouped.append(
            dict(prediction=str(classes[idx]), expected=next(iter(truth)), score=float(scores[idx]))
        )
    result = {}
    for label in classes:
        pred = [r for r in grouped if r["prediction"] == label]
        valid = []
        for threshold in sorted({r["score"] for r in pred}):
            selected = [r for r in pred if r["score"] >= threshold]
            correct = sum(r["expected"] == label for r in selected)
            if len(selected) >= min_groups and correct / len(selected) >= target_precision:
                valid.append(
                    dict(
                        threshold=threshold,
                        groups=len(selected),
                        correct=correct,
                        training_oof_precision=correct / len(selected),
                    )
                )
        result[str(label)] = (
            valid[0]
            if valid
            else dict(
                threshold=None,
                groups=0,
                reason="Insufficient independent OOF support at target precision",
            )
        )
    return result


def fit(rows, x, c):
    group_counts = Counter(r["group"] for r in rows)
    y = np.array([r["model"] for r in rows])
    weights = np.array([1.0 / group_counts[r["group"]] for r in rows])
    for label in set(y):
        weights[y == label] /= weights[y == label].sum()
    weights *= len(rows) / weights.sum()
    return LogisticRegression(C=c, max_iter=3000).fit(x, y, sample_weight=weights)


def train(rows, vectors, tune=True, min_groups=8):
    x = np.stack([vectors[r["key"]] for r in rows])
    y = np.array([r["model"] for r in rows])
    groups = np.array([r["group"] for r in rows])
    classes = np.unique(y)
    folds = list(StratifiedGroupKFold(3, shuffle=True, random_state=911).split(x, y, groups))
    candidates = []
    for c in [0.03, 0.1, 0.3, 1.0, 3.0] if tune else [1.0]:
        probabilities = np.zeros((len(rows), len(classes)))
        for a, b in folds:
            fold = fit([rows[i] for i in a], x[a], c)
            p = fold.predict_proba(x[b])
            for i, label in enumerate(fold.classes_):
                probabilities[b, list(classes).index(label)] = p[:, i]
        prediction = classes[probabilities.argmax(axis=1)]
        score = balanced_accuracy_score(y, prediction)
        candidates.append((score, c, probabilities))
    score, c, probabilities = max(candidates, key=lambda v: (v[0], -v[1]))
    model = fit(rows, x, c)
    records = [
        dict(group=r["group"], expected=r["model"], probabilities=p.tolist())
        for r, p in zip(rows, probabilities, strict=True)
    ]
    thresholds = calibrate(records, classes, min_groups=min_groups, target_precision=0.95)
    return serialize(model), dict(
        c=c,
        grouped_oof_balanced_accuracy=score,
        thresholds=thresholds,
        groups=len(set(groups)),
        examples=len(rows),
        candidates=[dict(c=v[1], score=v[0]) for v in candidates],
        oof_records=records,
    )
