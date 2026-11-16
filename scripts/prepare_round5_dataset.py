"""Freeze reviewed photos before model fitting, preserving all earlier controls."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts.audit_round4_candidates import canonical_source
from scripts.train_field_model_pilot import assert_independent

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/continuation_round5"
FUTURE = {"dt_75", "t_150k", "mtz_1523", "acros_550"}


def split_sources(rows):
    by = defaultdict(set)
    for r in rows:
        by[r["model"]].add(r["group"])
    held = set()
    for label, groups in by.items():
        ordered = sorted(
            groups, key=lambda g: hashlib.sha256(("509|" + label + "|" + g).encode()).hexdigest()
        )
        n = (
            min(len(ordered) - 1, max(2 if label in FUTURE else 1, round(len(ordered) * 0.25)))
            if len(ordered) > 1
            else 0
        )
        held.update(ordered[:n])
    return [dict(r, role="test" if r["group"] in held else "train") for r in rows]


def main():
    audit = json.loads((OUT / "audit.json").read_text(encoding="utf-8"))
    labels = json.loads((OUT / "manual_labels.json").read_text(encoding="utf-8"))
    reviewed = []
    eligible = []
    for r in audit:
        review = labels.get(r["id"])
        if r["decision"] == "needs_manual_review" and review is None:
            raise ValueError("Unreviewed photo " + r["id"])
        record = dict(r, manual=review)
        reviewed.append(record)
        if r["decision"] != "needs_manual_review" or not review or review["decision"] != "accepted":
            continue
        source = canonical_source(r["source"])
        eligible.append(
            dict(
                key="round5/" + r["id"],
                group="source-" + hashlib.sha256(source.encode()).hexdigest()[:12],
                model=review["model"],
                path=str((OUT / r["file"]).relative_to(ROOT)),
                sha256=r["sha256"],
                condition=review["condition"],
                source=r["source"],
                review_reason=review["reason"],
            )
        )
    selected = split_sources(eligible)
    freeze = OUT / "reviewed_split.json"
    if freeze.exists() and json.loads(freeze.read_text(encoding="utf-8")) != selected:
        raise ValueError("Frozen round5 split changed; start a new experiment")
    freeze.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "reports/round5_photo_review.json").write_text(
        json.dumps(reviewed, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    old = json.loads(
        (ROOT / "output/continuation_round4/field_pilot/dataset_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    train = old["train"] + [
        dict(r, cohort="round5_train") for r in selected if r["role"] == "train"
    ]
    test = old["test"] + [
        dict(
            r,
            cohort="round5_"
            + (
                "future"
                if r["model"] in FUTURE
                else "targets" if r["model"] in {"mtz_80", "mtz_820", "nh_t5000"} else "other"
            ),
        )
        for r in selected
        if r["role"] == "test"
    ]
    assert_independent(train, test)
    result = dict(train=train, test=test)
    (OUT / "dataset_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = dict(
        candidates=len(audit),
        decoded=sum("sha256" in r for r in audit),
        reviewed=len(labels),
        eligible=len(selected),
        audit=dict(Counter(r["decision"] for r in audit)),
        manual=dict(Counter(r["decision"] for r in labels.values())),
        new_by_class={
            label: dict(
                total=sum(r["model"] == label for r in selected),
                groups=len({r["group"] for r in selected if r["model"] == label}),
                train=sum(r["model"] == label and r["role"] == "train" for r in selected),
                test=sum(r["model"] == label and r["role"] == "test" for r in selected),
            )
            for label in sorted({r["model"] for r in selected})
        },
        train=len(train),
        test=len(test),
        condition=dict(Counter(r["condition"] for r in selected)),
    )
    (OUT / "data_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
