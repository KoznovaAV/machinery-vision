"""Audit new photo candidates against all frozen training and control data."""

import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from scripts.audit_round4_candidates import canonical_source, distance, fingerprint

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/continuation_round5"


def main():
    old = json.loads(
        (ROOT / "output/continuation_round4/field_pilot/dataset_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    refs = []
    seen = set()
    sources = set()
    cache_path = OUT / "reference_fingerprints.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    paths = [ROOT / r["path"] for s in old.values() for r in s]
    for folder in [
        "output/continuation_round4/photos",
        "output/continuation_round4/mtz1523_photos",
        "output/new_field_models/photos",
    ]:
        paths += list((ROOT / folder).glob("*"))
    for p in paths:
        if not p.is_file():
            continue
        signature = str(p.stat().st_size) + ":" + str(p.stat().st_mtime_ns)
        key = str(p.relative_to(ROOT))
        try:
            saved = cache.get(key, {})
            fp = saved["fingerprint"] if saved.get("signature") == signature else fingerprint(p)
            cache[key] = dict(signature=signature, fingerprint=fp)
        except (OSError, ValueError):
            continue
        if fp["sha256"] not in seen:
            refs.append(dict(fp, path=str(p.relative_to(ROOT))))
            seen.add(fp["sha256"])
    for name in [
        "reports/future_field_model_registry.json",
        "reports/axle_marking_new_control.json",
        "output/continuation_round4/new_control.json",
    ]:
        for r in json.loads((ROOT / name).read_text(encoding="utf-8")):
            if r.get("source"):
                sources.add(canonical_source(r["source"]))
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    for name in [
        "output/wet_round3/experiment_manifest.csv",
        "output/weather_expansion/round3_reviewed.csv",
        "output/mtz_model_pilot/reviewed.csv",
        "output/model_wave2/reviewed.csv",
        "output/mtz_expansion/reviewed_manifest.csv",
    ]:
        with (ROOT / name).open(encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row.get("source"):
                    sources.add(canonical_source(row["source"]))
    rows = []
    queue = json.loads((OUT / "queue.json").read_text(encoding="utf-8-sig"))
    for r in queue:
        row = dict(r, decision="unavailable", canonical_source=canonical_source(r["source"]))
        try:
            fp = fingerprint(OUT / r["file"])
        except (OSError, ValueError):
            rows.append(row)
            continue
        nearest = min((distance(fp, old), old["path"]) for old in refs)
        row.update(fp, nearest_prior=list(nearest), decision="needs_manual_review")
        if row["canonical_source"] in sources:
            row["decision"] = "prior_source"
        if fp["sha256"] in seen or nearest[0] <= 8:
            row["decision"] = "prior_duplicate"
        rows.append(row)
    contaminated = {
        r["canonical_source"] for r in rows if r["decision"] in {"prior_source", "prior_duplicate"}
    }
    for r in rows:
        if r["decision"] == "needs_manual_review" and r["canonical_source"] in contaminated:
            r["decision"] = "prior_source"
    fresh = [r for r in rows if r["decision"] == "needs_manual_review"]
    for i, r in enumerate(fresh):
        for prev in fresh[:i]:
            if distance(r, prev) <= 8:
                r.update(decision="new_duplicate", duplicate_of=prev["id"])
                break
    (OUT / "audit.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    reviewed = (
        json.loads((OUT / "manual_labels.json").read_text(encoding="utf-8"))
        if (OUT / "manual_labels.json").exists()
        else {}
    )
    candidates = [
        r for r in rows if r["decision"] == "needs_manual_review" and r["id"] not in reviewed
    ]
    for start in range(0, len(candidates), 9):
        sheet = Image.new("RGB", (1500, 1200), "white")
        d = ImageDraw.Draw(sheet)
        for j, r in enumerate(candidates[start : start + 9]):
            im = ImageOps.exif_transpose(Image.open(OUT / r["file"])).convert("RGB")
            im.thumbnail((495, 365))
            x = j % 3 * 500
            y = j // 3 * 400
            sheet.paste(im, (x, y + 28))
            d.text((x + 3, y + 3), r["id"] + " prior=" + str(r["nearest_prior"][0]), fill="black")
        sheet.save(OUT / f"pending-review-{start//9:02d}.jpg")
    print(
        "references", len(refs), "decisions", dict(Counter(r["decision"] for r in rows)), flush=True
    )


if __name__ == "__main__":
    main()
