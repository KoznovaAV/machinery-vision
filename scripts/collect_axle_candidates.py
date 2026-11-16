"""Collect source-linked photo candidates; never assign accepted labels automatically."""

import concurrent.futures
import csv
import json
import re
from pathlib import Path

import requests

ROOT = Path.cwd()
OUT = ROOT / "output/axle_round2"
OUT.mkdir(exist_ok=True)
sources = [
    ("mtz_820", "https://tractor.bg/product/belarus-mtz-mtz-820-2"),
    ("mtz_820", "https://tractor.bg/product/belarus-mtz-820-2"),
    ("mtz_80", "https://tractor.bg/product/belarus-mtz-80-1-90917"),
    ("mtz_80", "https://tractor.bg/product/belarus-mtz-mtz-80-43350"),
    ("mtz_80", "https://tractor.bg/product/belarus-mtz-80-8"),
    ("mtz_80", "https://tractor.bg/product/belarus-mtz-mtz-80-3"),
    ("mtz_80", "https://tractor.bg/product/belarus-mtz-80-47097"),
    ("mtz_80", "https://tractor.bg/product/belarus-mtz-mtz-80-11"),
]
old = list(
    csv.DictReader(
        open(ROOT / "output/weather_expansion/round3_reviewed.csv", encoding="utf-8-sig")
    )
)
oldgroups = {r["group"] for r in old}
oldurls = {r["source"].rstrip("/") for r in old}
queue = []


def collect(item):
    model, url = item
    try:
        res = requests.get(url, timeout=(4, 8))
        res.raise_for_status()
        html = res.text
        (OUT / ("page-" + str(sources.index(item)) + ".html")).write_text(html, encoding="utf-8")
        ids = re.findall(r'(?:products/pdf/|product_id["\s:=]+)(\d+)', html)
        group = "tractorbg-" + ids[0] if ids else "url-" + str(sources.index(item))
        gallery = sorted(set(re.findall(r'https://[^\s"<>]+/710x710\.jpg', html)))
        if group in oldgroups or url.rstrip("/") in oldurls:
            return [], dict(url=url, reason="previous_source_group", group=group)
        rows = [
            dict(
                id="axle2-" + str(sources.index(item)) + "-" + str(j),
                file="photos/" + f"axle2-{sources.index(item)}-{j}.jpg",
                url=u,
                source=url,
                group=group,
                query_model=model,
            )
            for j, u in enumerate(gallery[:5])
        ]
        return rows, dict(url=url, group=group, candidates=len(rows))
    except Exception as e:
        return [], dict(url=url, error=type(e).__name__)


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(collect, sources))
for rows, _ in results:
    queue.extend(rows)
(OUT / "queue.json").write_text(json.dumps(queue, indent=2), encoding="utf-8")
(OUT / "source_status.json").write_text(
    json.dumps([s for _, s in results], indent=2), encoding="utf-8"
)
print(json.dumps([s for _, s in results], indent=2))
