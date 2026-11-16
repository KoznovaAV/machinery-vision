"""Audit new candidate photos against frozen splits and previously reviewed sources."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import imagehash
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/continuation_round4"


def canonical_source(url):
    p = urlsplit(url)
    query = [
        (k, v)
        for k, v in parse_qsl(p.query, keep_blank_values=True)
        if not k.lower().startswith("utm_")
    ]
    return urlunsplit(
        (
            "https",
            p.netloc.lower().removeprefix("www."),
            p.path.rstrip("/"),
            urlencode(sorted(query)),
            "",
        )
    )


def fingerprint(path):
    with Image.open(path) as im:
        im.load()
        im = ImageOps.exif_transpose(im).convert("RGB")
        return dict(
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            phash=int(str(imagehash.phash(im)), 16),
            mirror=int(str(imagehash.phash(ImageOps.mirror(im))), 16),
            dimensions=list(im.size),
        )


def distance(a, b):
    return min(
        (x ^ y).bit_count() for x in [a["phash"], a["mirror"]] for y in [b["phash"], b["mirror"]]
    )


def main():
    previous = []
    sources = set()
    manifests = [
        "output/wet_round3/experiment_manifest.csv",
        "output/weather_expansion/round3_reviewed.csv",
        "output/mtz_model_pilot/reviewed.csv",
        "output/model_wave2/reviewed.csv",
        "output/mtz_expansion/reviewed_manifest.csv",
    ]
    for name in manifests:
        with (ROOT / name).open(encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("source"):
                    sources.add(canonical_source(row["source"]))
                path = row.get("path")
                if path and (ROOT / path).is_file():
                    previous.append(ROOT / path)
    for name, folder in [
        ("reports/axle_marking_new_control.json", ""),
        ("reports/new_field_models_collection.json", "output/new_field_models"),
    ]:
        for row in json.loads((ROOT / name).read_text(encoding="utf-8")):
            sources.add(canonical_source(row["source"]))
            path = ROOT / row["path"] if row.get("path") else ROOT / folder / row["file"]
            if path.is_file():
                previous.append(path)
    previous += list((ROOT / "output/open_set/attachment_control").glob("*.jpg"))
    refs = []
    seen = set()
    for path in previous:
        try:
            fp = fingerprint(path)
        except (OSError, ValueError):
            continue
        if fp["sha256"] in seen:
            continue
        seen.add(fp["sha256"])
        refs.append(dict(**fp, path=str(path.relative_to(ROOT))))
    queue = json.loads((OUT / "queue.json").read_text(encoding="utf-8-sig"))
    rows = []
    for r in queue:
        path = (OUT / r["file"]).resolve()
        if not path.is_relative_to(OUT.resolve()):
            raise ValueError("Photo escaped collection directory")
        row = dict(r, canonical_source=canonical_source(r["source"]), decision="unavailable")
        try:
            fp = fingerprint(path)
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
        r["canonical_source"] for r in rows if r["decision"] in ["prior_source", "prior_duplicate"]
    }
    for row in rows:
        if row["decision"] == "needs_manual_review" and row["canonical_source"] in contaminated:
            row["decision"] = "prior_source"
    fresh = [r for r in rows if r["decision"] == "needs_manual_review"]
    for i, row in enumerate(fresh):
        for prior in fresh[:i]:
            if distance(row, prior) <= 8:
                row.update(decision="new_duplicate", duplicate_of=prior["id"])
                break
    (OUT / "audit.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    candidates = [r for r in rows if r["decision"] == "needs_manual_review"]
    for start in range(0, len(candidates), 9):
        sheet = Image.new("RGB", (1350, 1050), "white")
        d = ImageDraw.Draw(sheet)
        for j, r in enumerate(candidates[start : start + 9]):
            im = ImageOps.exif_transpose(Image.open(OUT / r["file"])).convert("RGB")
            im.thumbnail((445, 310))
            x = j % 3 * 450
            y = j // 3 * 350
            sheet.paste(im, (x, y + 28))
            d.text(
                (x + 2, y + 3),
                r["id"] + " " + r["query_model"] + " prior=" + str(r["nearest_prior"][0]),
                fill="black",
            )
        sheet.save(OUT / f"candidate-review-{start//9}.jpg")
    from collections import Counter

    print("references", len(refs), "decisions", dict(Counter(r["decision"] for r in rows)))
    print("fresh", dict(Counter(r["query_model"] for r in candidates)))


if __name__ == "__main__":
    main()
