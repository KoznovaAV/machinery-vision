"""Prepare new training-only photos while preserving every protected test row."""

import json
from pathlib import Path

from scripts.benchmark_grouped_recognition import digest
from scripts.train_field_model_pilot import assert_independent

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/continuation_round6"


def main():
    old = json.loads(
        (ROOT / "output/continuation_round5/dataset_manifest.json").read_text(encoding="utf-8")
    )
    train = list(old["train"])
    review = ROOT / "reports/round6_photo_review.json"
    for r in json.loads(review.read_text(encoding="utf-8")):
        if r["manual_decision"] != "accepted":
            continue
        if r["decision"] != "needs_manual_review":
            raise ValueError("Protected or duplicated input")
        train.append(
            dict(
                key="round6/" + r["id"],
                group=r["group"],
                model=r["model"],
                path=str((OUT / r["file"]).relative_to(ROOT)),
                sha256=r["sha256"],
                condition=r["condition"],
                source=r["source"],
                role="train",
            )
        )
    assert_independent(train, old["test"])
    data = dict(train=train, test=old["test"])
    p = OUT / "dataset_manifest.json"
    if p.exists() and json.loads(p.read_text(encoding="utf-8")) != data:
        raise ValueError("Frozen manifest changed")
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "train",
        len(train),
        "same test",
        len(data["test"]),
        "previous manifest",
        digest(ROOT / "output/continuation_round5/dataset_manifest.json"),
        flush=True,
    )


if __name__ == "__main__":
    main()
