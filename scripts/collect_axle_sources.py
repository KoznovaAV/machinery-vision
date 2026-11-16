import json
import re
from pathlib import Path

import requests

p = Path("output/axle_round2")
p.mkdir(exist_ok=True)
urls = [
    "https://tractor.bg/product/belarus-mtz-mtz-820-2",
    "https://tractor.bg/product/belarus-mtz-820-2",
    "https://tractor.bg/product/belarus-mtz-820-40262",
    "https://tractor.bg/product/belarus-mtz-820-4",
    "https://tractor.bg/products/pdf/90917",
    "https://tractor.bg/products/pdf/43350",
    "https://agro.ria.com/ru/sell-traktor-mtz-80-bilorus-350018043.html",
    "https://agro.ria.com/ru/sell-traktor-mtz-80-bilorus-350374139.html",
]
for i, u in enumerate(urls):
    try:
        r = requests.get(u, timeout=(4, 8))
        r.raise_for_status()
        (p / f"source-{i}.html").write_text(r.text, encoding="utf-8")
        images = sorted(
            set(
                re.findall(r"(?:https?:)?//[^\s\"<>]+?\.(?:jpg|jpeg|webp)(?:\?[^\s\"<>]*)?", r.text)
            )
        )
        print(i, u, images[:12], flush=True)
    except Exception as e:
        print(i, type(e).__name__, flush=True)
(p / "source_urls.json").write_text(json.dumps(urls, indent=2), encoding="utf-8")
