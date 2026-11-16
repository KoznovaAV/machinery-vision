import argparse
import html
import json
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

agent = "MachineryVisionDatasetAudit/0.2 (local photo classification research)"
prefix = "https://commons.wikimedia.org/w/api.php?"
titles = {
    "mtz_1221": [
        "Belarus 1221.JPG",
        "Traktor Belarus 1221.4.JPG",
        "Belarus 1221.2 (open day 2017) 1.jpg",
        "Belarus 1221.3 (open day 2017) 1.jpg",
        "Belarus MTZ-1221 3 tractor.jpg",
        "Belarus 1221.jpg",
        "MTZ-Belarus forestry tractor (1221.3).jpg",
        "L1221.4 forestry tractor 1.jpg",
    ],
    "mtz_82": [
        "Belarus 82 (new).jpg",
        "Belarus 82.1, 2025 Tahitótfalu.jpg",
        "MTZ-82 Pécs 2016 1.jpg",
        "Belarus-Rakaw-Belarus 82.1-1.jpg",
        "MTZ-82 mit Anhänger.jpg",
        "MTZ-82 tractor.JPG",
    ],
}


def get(url):
    with urlopen(Request(url, headers={"User-Agent": agent}), timeout=15) as r:
        return r.read(20_000_000)


def text(x):
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", x or "")).split())


def main() -> None:
    parser = argparse.ArgumentParser(description="Download licensed Commons MTZ photo candidates.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    photos = args.output / "collection"
    photos.mkdir(parents=True, exist_ok=True)
    rows = []
    queue = []
    for label, names in titles.items():
        request = {
            "action": "query",
            "format": "json",
            "formatversion": 2,
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|mime",
            "titles": "|".join("File:" + name for name in names),
        }
        result = json.loads(get(prefix + urlencode(request)))
        for page in result["query"]["pages"]:
            if "imageinfo" not in page:
                continue
            item = page["imageinfo"][0]
            m = item["extmetadata"]

            def read(key):
                return text(m.get(key, {}).get("value", ""))

            raw = urlsplit(item["url"])
            media = urlunsplit((raw.scheme, raw.netloc, raw.path, "", ""))
            if raw.hostname != "upload.wikimedia.org" or item["mime"] != "image/jpeg":
                continue
            queue.append(
                dict(
                    id=f"commons-{page['pageid']}",
                    url=media,
                    source=item["descriptionurl"],
                    model=label,
                    description=read("ImageDescription"),
                    author=read("Artist"),
                    license=read("LicenseShortName"),
                )
            )
    limit = 0
    for item in queue:
        path = photos / (item["id"] + ".jpg")
        row = {**item, "collection_status": "pending"}
        if limit < 2:
            try:
                if not path.exists():
                    payload = get(item["url"])
                    if not payload.startswith(b"\xff\xd8\xff"):
                        raise ValueError("Expected a JPEG response.")
                    path.write_bytes(payload)
                limit = 0
                row["collection_status"] = "downloaded"
            except HTTPError as e:
                row["collection_status"] = str(e.code)
                if e.code == 429:
                    limit += 1
                    retry = e.headers.get("Retry-After", "")
                    if retry.isdecimal() and int(retry) > 4:
                        limit = 2
                    else:
                        time.sleep(3)
            except (OSError, URLError, ValueError):
                row["collection_status"] = "unavailable"
        rows.append(row)
        print(row["id"], row["collection_status"], flush=True)
        (args.output / "collection_sources.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if limit < 2:
            time.sleep(1)


if __name__ == "__main__":
    main()
