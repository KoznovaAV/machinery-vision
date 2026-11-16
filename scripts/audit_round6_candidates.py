"""Audit round6 candidates against all prior pixels and protected test sources."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from scripts.audit_round4_candidates import canonical_source, distance, fingerprint

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/continuation_round6"


def main():
    r5 = ROOT / "output/continuation_round5"
    data = json.loads((r5 / "dataset_manifest.json").read_text(encoding="utf-8"))
    testsha = {r["sha256"] for r in data["test"]}
    trainsha = {r["sha256"]: r["group"] for r in data["train"]}
    refs = [
        dict(v["fingerprint"], path=k)
        for k, v in json.loads(
            (r5 / "reference_fingerprints.json").read_text(encoding="utf-8")
        ).items()
    ]
    sources = set()
    train_sources = {}
    metadata = []
    for name in [
        "reports/round5_photo_review.json",
        "reports/future_field_model_registry.json",
        "reports/round4_photo_review.json",
        "reports/round4_mtz1523_review.json",
        "reports/axle_marking_new_control.json",
    ]:
        metadata += json.loads((ROOT / name).read_text(encoding="utf-8"))
    for name in [
        "output/wet_round3/experiment_manifest.csv",
        "output/weather_expansion/round3_reviewed.csv",
        "output/mtz_model_pilot/reviewed.csv",
        "output/model_wave2/reviewed.csv",
        "output/mtz_expansion/reviewed_manifest.csv",
    ]:
        with (ROOT / name).open(encoding="utf-8-sig") as f:
            metadata += list(csv.DictReader(f))
    for r in metadata:
        if r.get("sha256") in testsha and r.get("source"):
            sources.add(canonical_source(r["source"]))
        if r.get("sha256") in trainsha and r.get("source"):
            train_sources.setdefault(canonical_source(r["source"]), set()).add(
                trainsha[r["sha256"]]
            )
    (OUT / "protected_sources.json").write_text(json.dumps(sorted(sources)), encoding="utf-8")
    for r in json.loads((r5 / "audit.json").read_text(encoding="utf-8")):
        if "phash" in r:
            refs.append(dict(r, path=str((r5 / r["file"]).relative_to(ROOT))))
    unique = {r["sha256"]: r for r in refs}
    refs = list(unique.values())
    rows = []
    for r in json.loads((OUT / "queue_all.json").read_text(encoding="utf-8")):
        source = canonical_source(r["source"])
        row = dict(r, decision="unavailable", canonical_source=source)
        try:
            fp = fingerprint(OUT / r["file"])
        except (OSError, ValueError):
            rows.append(row)
            continue
        nearest = min((distance(fp, old), old["path"]) for old in refs)
        row.update(fp, nearest_prior=list(nearest), decision="needs_manual_review")
        if source in sources:
            row["decision"] = "protected_control_source"
        if nearest[0] <= 8:
            row["decision"] = "prior_duplicate"
        if row["decision"] == "needs_manual_review":
            for prev in rows:
                if "phash" in prev and distance(fp, prev) <= 8:
                    row.update(decision="new_duplicate", duplicate_of=prev["id"])
                    break
        groups = train_sources.get(source, set())
        row["group"] = (
            next(iter(groups))
            if len(groups) == 1
            else "source-" + hashlib.sha256(source.encode()).hexdigest()[:12]
        )
        rows.append(row)
    (OUT / "audit.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    todo = [r for r in rows if r["decision"] == "needs_manual_review"]
    for start in range(0, len(todo), 9):
        sheet = Image.new("RGB", (1500, 1200), "white")
        d = ImageDraw.Draw(sheet)
        for j, r in enumerate(todo[start : start + 9]):
            im = ImageOps.exif_transpose(Image.open(OUT / r["file"])).convert("RGB")
            im.thumbnail((495, 360))
            x = j % 3 * 500
            y = j // 3 * 400
            sheet.paste(im, (x, y + 30))
            d.text((x, y), r["id"], fill="black")
        sheet.save(OUT / f"review-{start//9:02d}.jpg")
    print(dict(Counter(r["decision"] for r in rows)), flush=True)


if __name__ == "__main__":
    main()
