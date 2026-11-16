"""Extract matching features and colour-invariant training views for weather round 2."""

import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from scripts.benchmark_grouped_recognition import digest, embeddings, encoder
from scripts.benchmark_tractor_localization import crop, load_detector, tractor_box
from scripts.predict_tractor_conditions import CONVNEXT_WEIGHTS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/winter_round2"
CROP_DIRS = (
    "output/winter_round2/views", "output/mvp/views", "output/tractor_parts/views",
    "output/continuation_round5/views", "output/continuation_round6/views",
)


def crop_path(row):
    for folder in CROP_DIRS:
        for name in ("machine.jpg", "tractor.jpg"):
            path = ROOT / folder / row["sha256"] / name
            if path.is_file():
                return path
    return None


def fused(files):
    a = embeddings(encoder("dinov2", ROOT / "weights/dinov2-small.safetensors"), files, True)
    gc.collect()
    b = embeddings(encoder("convnext", CONVNEXT_WEIGHTS), files, True)
    return {k: np.concatenate((a[k], b[k])) for k in files}


def main():
    torch.set_num_threads(5)
    data = json.loads((OUT / "dataset_manifest.json").read_bytes())
    new = [r for r in data["new_review"] if r["decision"] == "accepted"]
    needed = [r for r in new if crop_path(r) is None]
    if needed:
        detector = load_detector(ROOT / "weights/grounding-dino-tiny")
        for i, row in enumerate(needed):
            with Image.open(ROOT / row["path"]) as handle:
                image = ImageOps.exif_transpose(handle).convert("RGB")
            detection = tractor_box(image, detector, 320, 544).get("foreground")
            machine = crop(image, detection["box"]) if detection else image
            destination = OUT / "views" / row["sha256"] / "machine.jpg"
            destination.parent.mkdir(parents=True, exist_ok=True)
            machine.save(destination, quality=95)
            print("crop",i+1,"/",len(needed),row["key"],flush=True)
        del detector
        gc.collect()
    files = {}
    for row in new:
        files["full/" + row["key"]] = ROOT / row["path"]
        files["crop/" + row["key"]] = crop_path(row)
    signature = hashlib.sha256("|".join(k+":"+digest(p) for k,p in sorted(files.items())).encode()).hexdigest()
    cache = OUT / "features.npz"
    if not cache.exists() or str(np.load(cache)["signature"]) != signature:
        values = fused(files)
        np.savez_compressed(cache,signature=signature,keys=list(values),values=list(values.values()))
    print("RGB features ready",len(files),flush=True)

    rows = data["train"] + data["family_only_train"]
    signature = hashlib.sha256(
        "|".join(r["key"]+":"+r["sha256"]+":"+digest(crop_path(r)) for r in rows).encode()).hexdigest()
    cache = OUT / "gray_features.npz"
    if cache.exists() and str(np.load(cache)["signature"]) == signature:
        print("Grayscale cache ready",flush=True)
        return
    gray = {}
    for row in rows:
        for view, path in (("full", ROOT / row["path"]), ("crop", crop_path(row))):
            with Image.open(path) as handle:
                image = ImageOps.exif_transpose(handle).convert("RGB")
                image = ImageOps.pad(image,(224,224),method=Image.Resampling.BICUBIC,
                                     color=(124,116,104))
                image = ImageOps.grayscale(image).convert("RGB")
            gray[view + "/" + row["key"]] = image
    print("Extracting grayscale views",len(gray),flush=True)
    values = fused(gray)
    np.savez_compressed(cache,signature=signature,keys=list(values),values=list(values.values()))
    print("Grayscale features ready",len(values),flush=True)


if __name__ == "__main__":
    main()
