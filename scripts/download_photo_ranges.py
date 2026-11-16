"""Resume public dataset photo downloads using bounded HTTP byte ranges."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlsplit

import requests
from PIL import Image

BLOCKED = set()
LOCK = threading.Lock()


def range_total(response, start):
    if response.status_code != 206:
        raise ValueError("Server did not honor a byte range.")
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
    if not match:
        raise ValueError("Missing or invalid Content-Range.")
    first, last, total = map(int, match.groups())
    if first != start or last < first or last >= total or len(response.content) != last - first + 1:
        raise ValueError("Incomplete or noncontiguous range response.")
    return total


def complete(path):
    try:
        with Image.open(path) as image:
            image.load()
        return True
    except (OSError, ValueError):
        return False


def download(row, directory):
    path = (directory / row["file"]).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("Photo path escaped collection directory.")
    if complete(path):
        return dict(id=row["id"], status="already_complete")
    host = urlsplit(row["url"]).hostname
    with LOCK:
        if host in BLOCKED:
            return dict(id=row["id"], status="host_deferred")
    session = requests.Session()
    session.headers["User-Agent"] = "MachineryVisionDataset/0.3 (local research)"
    try:
        response = session.get(row["url"], headers={"Range": "bytes=0-16383"}, timeout=(4, 5))
        if response.status_code in {403, 429}:
            with LOCK:
                BLOCKED.add(host)
            return dict(
                id=row["id"],
                status=f"http_{response.status_code}",
                retry_after=response.headers.get("Retry-After", ""),
            )
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        if response.status_code == 200:
            if len(response.content) > 5_000_000:
                raise ValueError("Full response exceeded photo limit.")
            path.write_bytes(response.content)
        else:
            total = range_total(response, 0)
            if total > 2_100_000:
                raise ValueError("Photo exceeded range download limit.")
            previous = path.read_bytes() if path.exists() else b""
            prefix = response.content
            if not previous or previous[: len(prefix)] != prefix or len(previous) > total:
                path.write_bytes(prefix)
            elif len(previous) < len(prefix):
                path.write_bytes(prefix)
            cursor = path.stat().st_size
            etag = response.headers.get("ETag") or response.headers.get("Last-Modified")
            with path.open("ab") as handle:
                while cursor < total:
                    with LOCK:
                        if host in BLOCKED:
                            return dict(id=row["id"], status="host_deferred_partial")
                    headers = {"Range": f"bytes={cursor}-{min(total - 1, cursor + 16383)}"}
                    if etag:
                        headers["If-Range"] = etag
                    part = session.get(row["url"], headers=headers, timeout=(4, 5))
                    if part.status_code in {403, 429}:
                        with LOCK:
                            BLOCKED.add(host)
                        return dict(
                            id=row["id"],
                            status=f"http_{part.status_code}",
                            retry_after=part.headers.get("Retry-After", ""),
                        )
                    part.raise_for_status()
                    if range_total(part, cursor) != total:
                        raise ValueError("Source photo changed during download.")
                    handle.write(part.content)
                    cursor += len(part.content)
        if not complete(path):
            raise ValueError("Downloaded image did not fully decode.")
        return dict(
            id=row["id"],
            status="downloaded",
            bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    except (OSError, ValueError, requests.RequestException) as error:
        return dict(id=row["id"], status="unavailable", reason=str(error)[:160])
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=3)
    parser.add_argument("--defer-host", action="append", default=[])
    args = parser.parse_args()
    BLOCKED.update(args.defer_host)
    rows = json.loads(args.queue.read_text(encoding="utf-8-sig"))
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(download, row, args.directory) for row in rows]
        for future in as_completed(pending):
            result = future.result()
            results.append(result)
            print(result["id"], result["status"], flush=True)
            (args.directory / "range_status.json").write_text(
                json.dumps(results, indent=2) + "\n", encoding="utf-8"
            )


if __name__ == "__main__":
    main()
