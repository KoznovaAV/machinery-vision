"""Fetch the upstream weights required by the current machinery bundle."""

import hashlib
import json
import os
import shutil
from pathlib import Path
from urllib.request import urlopen

from src.config.hierarchy_artifacts import ROOT, hierarchy_bundle_path

DINO = (
    "https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m/"
    "resolve/bdc8408/model.safetensors"
)
DETECTOR = (
    "https://huggingface.co/IDEA-Research/grounding-dino-tiny/"
    "resolve/a2bb814dd30d776dcf7e30523b00659f4f141c71"
)
CONVNEXT = "https://download.pytorch.org/models/convnext_tiny-983f1562.pth"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ensure_file(url, path, expected=None):
    if path.is_file() and (expected is None or digest(path) == expected):
        print(f"Ready: {path.name}", flush=True)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".download")
    print(f"Downloading: {path.name}", flush=True)
    try:
        with urlopen(url, timeout=90) as response, partial.open("wb") as stream:
            shutil.copyfileobj(response, stream, length=1024 * 1024)
        if expected is not None and digest(partial) != expected:
            raise ValueError(f"Unexpected SHA-256: {path.name}")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def main():
    from torch.hub import get_dir

    if not os.environ.get("TORCH_HOME"):
        raise SystemExit("Set TORCH_HOME to the desired cache directory before downloading.")
    bundle = json.loads(hierarchy_bundle_path().read_text(encoding="utf-8"))
    weights = ROOT / "weights"
    detector = weights / "grounding-dino-tiny"
    ensure_file(DINO, weights / "dinov2-small.safetensors", bundle["encoder_sha256"])
    ensure_file(
        CONVNEXT,
        Path(get_dir()) / "checkpoints/convnext_tiny-983f1562.pth",
        bundle["secondary_encoder_sha256"],
    )
    ensure_file(
        f"{DETECTOR}/model.safetensors",
        detector / "model.safetensors",
        bundle["detector_sha256"],
    )
    for name in (
        "added_tokens.json",
        "config.json",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
    ):
        ensure_file(f"{DETECTOR}/{name}", detector / name)


if __name__ == "__main__":
    main()
