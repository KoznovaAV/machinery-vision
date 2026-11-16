"""Maintain immutable source-group splits for candidate future model classes."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from scripts.audit_round4_candidates import canonical_source

ROOT = Path(__file__).resolve().parents[1]


def preserve_group_splits(rows, previous):
    """Existing controls cannot become training data when new sources arrive."""
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Repeated candidate id")
    prior_by_id = {r["id"]: r for r in previous}
    if set(prior_by_id) - {r["id"] for r in rows}:
        raise ValueError("Registered candidate missing from updated corpus")
    group_splits = {}
    for r in previous:
        existing = group_splits.setdefault(r["group"], r["split"])
        if existing != r["split"]:
            raise ValueError("Source group has conflicting splits")
    for r in rows:
        prior = prior_by_id.get(r["id"])
        if prior and any(prior[k] != r[k] for k in ("sha256", "model", "group")):
            raise ValueError("Previously registered candidate changed")
    if not previous:
        by = defaultdict(set)
        for r in rows:
            by[r["model"]].add(r["group"])
        for label, groups in by.items():
            ordered = sorted(
                groups, key=lambda g: hashlib.sha256(("83|" + label + "|" + g).encode()).hexdigest()
            )
            held = set(ordered[: max(1, round(len(ordered) * 0.2))])
            for g in groups:
                group_splits[g] = "future_test" if g in held else "future_train_pool"
    result = []
    for r in rows:
        if r["group"] not in group_splits:
            group_splits[r["group"]] = (
                "future_test"
                if int(hashlib.sha256(("83|" + r["group"]).encode()).hexdigest(), 16) % 5 == 0
                else "future_train_pool"
            )
        result.append(dict(r, split=group_splits[r["group"]]))
    return result


def main():
    path = ROOT / "reports/future_field_model_registry.json"
    previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    rows = []
    for r in json.loads(
        (ROOT / "reports/new_field_models_collection.json").read_text(encoding="utf-8")
    ):
        if r["decision"] != "reviewed_candidate":
            continue
        rows.append(
            dict(
                id=r["id"],
                model=r["model"],
                group="source-"
                + hashlib.sha256(canonical_source(r["source"]).encode()).hexdigest()[:12],
                source=r["source"],
                path="output/new_field_models/" + r["file"],
                sha256=r["sha256"],
            )
        )
    for name in ["reports/round4_photo_review.json", "reports/round4_mtz1523_review.json"]:
        file = ROOT / name
        if not file.exists():
            continue
        for r in json.loads(file.read_text(encoding="utf-8")):
            if r["decision"] != "reviewed_future_candidate":
                continue
            rows.append(
                dict(
                    id=r["id"],
                    model=r["model"],
                    group=r["group"],
                    source=r["source"],
                    path="output/continuation_round4/" + r["file"],
                    sha256=r["sha256"],
                )
            )
    result = preserve_group_splits(rows, previous)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    from collections import Counter

    print(dict(Counter(r["model"] for r in result)))


if __name__ == "__main__":
    main()
