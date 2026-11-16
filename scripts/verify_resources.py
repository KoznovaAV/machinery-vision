"""Verify the resource bundle and its compatibility with the local model artifacts."""

import hashlib
import json
from pathlib import Path

from src.config.runtime import ROOT


def verify(root=ROOT):
    root = Path(root).resolve()
    manifest = json.loads((root / "RESOURCES_MANIFEST.json").read_bytes())
    if (
        manifest.get("format") != "machinery-vision-resources-v1"
        or not manifest.get("files")
        or not manifest.get("compatible_artifacts")
    ):
        raise ValueError("Invalid resource manifest")
    errors = []
    records = manifest["files"] + manifest["compatible_artifacts"]
    for record in records:
        path = (root / record["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            errors.append("Missing or external path: " + record["path"])
            continue
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != record["sha256"] or path.stat().st_size != record["bytes"]:
            errors.append("Changed: " + record["path"])
    return {"passed": not errors, "files": len(manifest["files"]), "errors": errors}


def main():
    try:
        result = verify()
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit("Cannot verify resources: " + str(error)) from None
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(not result["passed"])


if __name__ == "__main__":
    main()
