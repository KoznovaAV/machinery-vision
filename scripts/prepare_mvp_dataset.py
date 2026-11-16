"""Build the MVP manifest from explicit visual decisions and protected source groups."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps
import imagehash

from scripts.audit_round4_candidates import canonical_source

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/mvp"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprints(path):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return [int(str(imagehash.phash(view)), 16)
                for view in (image, ImageOps.mirror(image))]


def condition(dirty=None, snow=0, rain=0, wet=None):
    return dict(dirty=dirty, snow=snow, rain=rain, wet=wet)


def main():
    prior = json.loads((ROOT / "output/continuation_round6/hierarchy_dataset_manifest.json").read_bytes())
    protected = prior["test"]
    old_rows = prior["train"] + prior.get("family_only_train", []) + protected
    exact = {r["sha256"] for r in old_rows}
    old_fingerprints = [(r["sha256"], fingerprints(ROOT / r["path"])) for r in old_rows]
    test_hashes = {r["sha256"] for r in protected}
    test_phash = [f for h, f in old_fingerprints if h in test_hashes]
    source_roles = {}
    for name in (
        "output/weather_expansion/round3_reviewed.csv",
        "output/targeted_weather2/reviewed.csv",
        "output/mtz_expansion/reviewed_manifest.csv",
    ):
        with (ROOT / name).open(encoding="utf-8-sig") as handle:
            for r in csv.DictReader(handle):
                if r.get("source") and r.get("sha256") in test_hashes:
                    source_roles[canonical_source(r["source"])] = "test"
    for r in protected:
        if r.get("source"):
            source_roles[canonical_source(r["source"])] = "test"

    decisions = json.loads((OUT / "visual_decisions.json").read_bytes())
    downloads = json.loads((OUT / "downloaded.json").read_bytes())
    reviewed = []
    accepted = []
    seen = list(old_fingerprints)
    for row in downloads:
        decision = decisions.get(row["id"])
        if row["status"] != "downloaded" or decision is None:
            continue
        record = dict(row, **decision)
        record["key"] = "mvp/" + row["id"]
        record["source"] = canonical_source(row["source"])
        record["group"] = "source-" + hashlib.sha256(record["source"].encode()).hexdigest()[:16]
        path = ROOT / row["path"]
        assert sha(path) == row["sha256"]
        fp = fingerprints(path)
        record["review_method"] = "visual_contact_sheet_and_source_identity"
        if decision["decision"] == "accepted":
            nearest = min((min((a ^ b).bit_count() for a in fp for b in old), h)
                          for h, old in seen)
            if record["source"] in source_roles or row["sha256"] in exact or nearest[0] <= 6:
                record.update(decision="duplicate_or_protected_source", nearest_sha256=nearest[1])
            else:
                accepted.append(record)
                seen.append((row["sha256"], fp))
                exact.add(row["sha256"])
        reviewed.append(record)

    condition_rows = []
    for row in reviewed:
        if row["decision"] == "accepted" and row.get("condition") is not None:
            condition_rows.append(dict(row, role="train"))
    weather = json.loads((OUT / "prior_weather.json").read_bytes())
    for idx, attrs in json.loads((OUT / "prior_weather_decisions.json").read_bytes()).items():
        r = weather[int(idx)]
        fp = fingerprints(ROOT / r["path"])
        protected_image = r["sha256"] in test_hashes or any(
            min((a ^ b).bit_count() for a in fp for b in old) <= 6 for old in test_phash)
        condition_rows.append(dict(
            key="prior_weather/" + idx, group=r["group"], path=r["path"], sha256=r["sha256"],
            source=r["source"], condition=attrs,
            role="test" if protected_image or "test" in r["split"] or "outside" in r["split"] else "train",
        ))

    base = json.loads((OUT / "base_photos.json").read_bytes())
    excluded_base = {4, 5, 9, 13, 15, 22, 28}
    dirty_base = {0, 1, 2, 3, 6, 8, 10, 11, 12, 14, 16}
    for i, name in enumerate(base):
        if i in excluded_base:
            continue
        path = ROOT / name
        condition_rows.append(dict(
            key="real_state/" + str(i), group="real-state-" + sha(path)[:16],
            path=name, sha256=sha(path), source="local_real_state",
            condition=condition(int(i >= 17 or i in dirty_base)),
            role="test",
        ))

    # All close images and source groups inherit test membership.
    extra_control = {"mvp/m047", "mvp/m098", "mvp/m111", "mvp/m117", "mvp/m151",
                     "mvp/n003", "mvp/n004", "mvp/n010", "mvp/m027"}
    for r in condition_rows:
        if r["key"] in extra_control:
            r["role"] = "test"
    test_groups = {r["group"] for r in condition_rows if r["role"] == "test"}
    control_fps = [(r["sha256"], fingerprints(ROOT / r["path"]))
                   for r in condition_rows if r["role"] == "test"]
    for r in condition_rows:
        if r["group"] in test_groups:
            r["role"] = "test"
        elif r["role"] == "train":
            fp = fingerprints(ROOT / r["path"])
            if any(r["sha256"] == h or min((a ^ b).bit_count() for a in fp for b in ph) <= 6
                   for h, ph in control_fps):
                r["role"] = "test"

    unique = {}
    for r in condition_rows:
        if r["sha256"] not in unique or r["role"] == "test":
            unique[r["sha256"]] = r
    condition_rows = list(unique.values())

    manifest = dict(
        train=prior["train"] + [dict(r, role="train") for r in accepted if r.get("model")],
        family_only_train=prior.get("family_only_train", []) +
            [dict(r, role="train") for r in accepted if r.get("family") and not r.get("model")],
        test=protected,
        condition_rows=condition_rows,
        new_review=reviewed,
        region_negatives=[r for r in accepted if r.get("region_label") == "attachment"],
    )
    (OUT / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    print("new", len(accepted), Counter(r.get("model") or r.get("family") or r["target"] for r in accepted))
    print("conditions", Counter(r["role"] for r in condition_rows))
    print("attributes", {k: dict(Counter(r["condition"].get(k) for r in condition_rows))
                         for k in ("dirty", "snow", "rain", "wet")})


if __name__ == "__main__":
    main()
