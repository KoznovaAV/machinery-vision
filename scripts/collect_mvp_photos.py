"""Download a bounded candidate queue and render sheets for manual review."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageOps

OUT = Path(__file__).resolve().parents[1] / "output/mvp"


def collect(row):
    record = dict(row)
    path = OUT / "photos" / (row["id"] + ".jpg")
    try:
        if not path.exists():
            with requests.get(row["url"], timeout=(4, 8), stream=True) as response:
                response.raise_for_status()
                payload = bytearray()
                for chunk in response.iter_content(65536):
                    payload.extend(chunk)
                    if len(payload) > 6000000:
                        raise ValueError("Image byte limit")
            with Image.open(BytesIO(payload)) as handle:
                handle.load()
                if min(handle.size) < 180:
                    raise ValueError("Small image")
            path.write_bytes(payload)
        with Image.open(path) as handle:
            handle.load()
        record.update(path=path.relative_to(OUT.parents[1]).as_posix(),
                      sha256=hashlib.sha256(path.read_bytes()).hexdigest(), status="downloaded")
    except (OSError, ValueError, requests.RequestException) as exc:
        record.update(status="failed", reason=type(exc).__name__)
    return record


def sheets(rows):
    for target in sorted({r["target"] for r in rows}):
        batch = [r for r in rows if r["target"] == target and r["status"] == "downloaded"]
        if not batch:
            continue
        sheet = Image.new("RGB", (1000, 200 * ((len(batch) + 3) // 4)), "white")
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(batch):
            x, y = (i % 4) * 250, (i // 4) * 200
            with Image.open(OUT.parents[1] / row["path"]) as handle:
                thumb = ImageOps.contain(ImageOps.exif_transpose(handle).convert("RGB"), (244, 173))
            sheet.paste(thumb, (x + (250 - thumb.width) // 2, y))
            draw.text((x + 6, y + 176), row["id"], fill="black")
        sheet.save(OUT / (target + ".jpg"), quality=90)


def main():
    (OUT / "photos").mkdir(exist_ok=True)
    queue = json.loads((OUT / "queue.json").read_bytes())
    previous = OUT / "downloaded.json"
    cached = {r["id"]: r for r in json.loads(previous.read_bytes())} if previous.exists() else {}
    pending = [r for r in queue if r["id"] not in cached]
    with ThreadPoolExecutor(max_workers=8) as pool:
        cached.update({r["id"]: r for r in pool.map(collect, pending)})
    rows = [cached[r["id"]] for r in queue]
    (OUT / "downloaded.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    sheets(rows)
    print(sum(r["status"] == "downloaded" for r in rows), "/", len(rows), flush=True)


if __name__ == "__main__":
    main()
