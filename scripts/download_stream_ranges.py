"""Bounded streaming HTTP ranges with source validators and persistent per-queue logs."""

import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from urllib3.exceptions import HTTPError as UrllibHTTPError

from scripts.download_photo_ranges import complete

LOCK = threading.Lock()
BLOCKED = set()


def read_part(session, url, start, max_bytes=2100000):
    response = session.get(
        url, headers={"Range": f"bytes={start}-{start+16383}"}, stream=True, timeout=(4, 8)
    )
    try:
        if response.status_code in {403, 429}:
            from urllib.parse import urlsplit

            with LOCK:
                BLOCKED.add(urlsplit(url).hostname)
            raise ValueError(f"HTTP {response.status_code}; host deferred")
        response.raise_for_status()
        if response.status_code == 206:
            match = re.fullmatch(
                r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", "")
            )
            if not match:
                raise ValueError("Invalid Content-Range")
            first, last, total = map(int, match.groups())
            if first != start or not (first <= last < total) or last - first + 1 > 16384:
                raise ValueError("Noncontiguous range")
            count = last - first + 1
        elif response.status_code == 200 and start == 0:
            count = int(response.headers.get("Content-Length", 0))
            total = count
            if not 0 < count <= max_bytes:
                raise ValueError("Unbounded full response")
        else:
            raise ValueError("Range not honored")
        if total > max_bytes:
            raise ValueError("Photo too large")
        payload = response.raw.read(count)
        if len(payload) != count:
            raise ValueError("Truncated range")
        return payload, total, response.headers.get("ETag"), response.headers.get("Last-Modified")
    finally:
        response.close()


def download(row, directory, max_bytes=2100000):
    from urllib.parse import urlsplit

    path = (directory / row["file"]).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("Escaped collection root")
    if complete(path):
        return dict(id=row["id"], status="already_complete")
    with LOCK:
        if urlsplit(row["url"]).hostname in BLOCKED:
            return dict(id=row["id"], status="host_deferred")
    session = requests.Session()
    session.headers["User-Agent"] = "MachineryVisionDataset/0.5 (local research)"
    try:
        first, total, etag, modified = read_part(session, row["url"], 0, max_bytes)
        path.parent.mkdir(parents=True, exist_ok=True)
        previous = path.read_bytes() if path.exists() else b""
        if not previous.startswith(first) or len(previous) > total:
            path.write_bytes(first)
        elif len(previous) < len(first):
            path.write_bytes(first)
        cursor = path.stat().st_size
        with path.open("ab") as handle:
            while cursor < total:
                with LOCK:
                    if urlsplit(row["url"]).hostname in BLOCKED:
                        raise ValueError("Host deferred")
                payload, size, newetag, newmodified = read_part(session, row["url"], cursor, max_bytes)
                if size != total or newetag != etag or newmodified != modified:
                    raise ValueError("Source validators changed")
                handle.write(payload)
                cursor += len(payload)
        if not complete(path):
            raise ValueError("Image did not decode completely")
        return dict(id=row["id"], status="downloaded", bytes=path.stat().st_size)
    except (OSError, ValueError, requests.RequestException, UrllibHTTPError) as e:
        return dict(id=row["id"], status="unavailable", reason=str(e)[:200])
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--max-photo-bytes", type=int, default=2100000)
    args = parser.parse_args()
    if not 1 <= args.max_photo_bytes <= 12000000:
        parser.error("--max-photo-bytes must be between 1 and 12000000")
    rows = json.loads(args.queue.read_text(encoding="utf-8-sig"))
    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(download, r, args.directory, args.max_photo_bytes) for r in rows]):
            result = future.result()
            results.append(result)
            print(result["id"], result["status"], flush=True)
            (args.directory / (args.queue.stem + "-status.json")).write_text(
                json.dumps(results, indent=2), encoding="utf-8"
            )


if __name__ == "__main__":
    main()
