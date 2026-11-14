"""Build a candidate family split from the reviewed inventory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

CONFIRMED_SERIES = (
    ("originals-chtz:145", "originals-chtz:122", "originals-chtz:076"),
    ("originals-chtz:131", "originals-chtz:050", "originals-chtz:041"),
    ("originals-johndeere:007", "originals-johndeere:125"),
    ("originals-kirovets:071", "originals-kirovets:041"),
    ("originals-mtz_belarus:004", "originals-mtz_belarus:236"),
    ("originals-mtz_belarus:223", "originals-mtz_belarus:244", "originals-mtz_belarus:282"),
    (
        "originals-mtz_belarus:146",
        "originals-mtz_belarus:172",
        "originals-mtz_belarus:258",
        "originals-mtz_belarus:315",
    ),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_family_rows(audit: Path, data: Path) -> list[dict]:
    review = {row["review_id"]: row for row in read_csv(audit / "visual_review.csv")}
    rows = []
    for record in read_csv(audit / "inventory.csv"):
        if record["tree"] != "processed":
            continue
        photo = data / record["path"]
        if digest(photo) != record["sha256"]:
            raise ValueError(f"Dataset inventory is out of date for {record['path']}")
        flags = review[record["review_id"]]["flags"]
        rows.append(
            {
                "path": record["path"],
                "sha256": record["sha256"],
                "family": record["family"],
                "original_split": record["split"],
                "review_id": record["review_id"],
                "hashes": (int(record["phash"], 16), int(record["phash_flip"], 16)),
                "review_flags": flags,
            }
        )
    if not rows:
        raise ValueError("No processed images in the inventory.")
    return rows


def group_rows(rows: list[dict], max_distance: int) -> None:
    parent = list(range(len(rows)))

    def root(i: int) -> int:
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for (i, a), (j, b) in itertools.combinations(enumerate(rows), 2):
        distance = min((x ^ y).bit_count() for x in a["hashes"] for y in b["hashes"])
        if a["sha256"] == b["sha256"] or distance <= max_distance:
            parent[root(j)] = root(i)
    by_review_id = {row["review_id"]: i for i, row in enumerate(rows)}
    for series in CONFIRMED_SERIES:
        present = [by_review_id[review_id] for review_id in series if review_id in by_review_id]
        if present:
            anchor = root(present[0])
            for member in present[1:]:
                parent[root(member)] = anchor

    for i, row in enumerate(rows):
        row["group"] = rows[root(i)]["review_id"]
        del row["hashes"]


def split_rows(rows: list[dict], seed: int) -> dict:
    accepted = []
    for row in rows:
        flags = row["review_flags"].split(";")
        row["split"] = "excluded" if "likely_game_render" in flags else ""
        if not row["split"]:
            accepted.append(row)

    families = np.asarray([row["family"] for row in accepted])
    groups = np.asarray([row["group"] for row in accepted])
    folds = list(
        StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(
            np.zeros(len(accepted)), families, groups
        )
    )
    target = np.asarray([0.6, 0.2, 0.2])
    class_ids, encoded = np.unique(families, return_inverse=True)
    totals = np.bincount(encoded, minlength=len(class_ids))

    def discrepancy(val: int, test: int) -> float:
        split_code = np.zeros(len(accepted), dtype=int)
        split_code[folds[val][1]] = 1
        split_code[folds[test][1]] = 2
        counts = np.array(
            [np.bincount(encoded[split_code == i], minlength=len(class_ids)) for i in range(3)]
        )
        if np.any(counts == 0):
            return float("inf")
        return float(np.square(counts / totals - target[:, None]).sum())

    val_fold, test_fold = min(itertools.combinations(range(5), 2), key=lambda x: discrepancy(*x))
    if not np.isfinite(discrepancy(val_fold, test_fold)):
        raise ValueError("The grouped data cannot put every family in each split.")
    val, test = set(folds[val_fold][1]), set(folds[test_fold][1])
    for i, row in enumerate(accepted):
        row["split"] = "val" if i in val else "test" if i in test else "train"

    split_names = ("train", "val", "test")
    members = {name: {r["group"] for r in accepted if r["split"] == name} for name in split_names}
    if any(left & right for left, right in itertools.combinations(members.values(), 2)):
        raise RuntimeError("Image groups overlap across splits.")
    return {
        "seed": seed,
        "files": dict(Counter(r["split"] for r in rows)),
        "groups_by_split": {name: len(members[name]) for name in split_names},
        "families": {
            name: dict(Counter(r["family"] for r in rows if r["split"] == name))
            for name in split_names
        },
        "mixed_family_groups": sum(
            len({r["family"] for r in accepted if r["group"] == group}) > 1 for group in set(groups)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--audit", type=Path, default=Path("output/dataset_audit"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phash-distance", type=int, default=8, choices=range(0, 17))
    args = parser.parse_args()
    rows = load_family_rows(args.audit, args.data)
    group_rows(rows, args.phash_distance)
    stats = {"phash_distance": args.phash_distance, **split_rows(rows, args.seed)}
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "family_split.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "family_split_summary.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
