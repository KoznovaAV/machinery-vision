"""Append reviewed new-image features while keeping frozen control features intact."""

import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from src.config.runtime import ROOT
from src.data.manifest import load_dataset, load_features, read, assert_independent, CONTROLS
from src.models.encoders import digest, embeddings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.parse_args()
    data = load_dataset()
    assert_independent(
        data["train"] + data["family_only_train"], [r for n in CONTROLS for r in data[n]]
    )
    full, cropped = load_features()
    index = read(ROOT / "datasets/features/index.json")
    rows = {
        r["key"]: r
        for n in ("train", "family_only_train", *CONTROLS, "condition_rows")
        for r in data[n]
    }
    pending = []
    for key, r in rows.items():
        path = (ROOT / r["path"]).resolve()
        if not path.is_relative_to(ROOT.resolve()) or digest(path) != r["sha256"]:
            raise ValueError("Invalid image: " + key)
        if key in index:
            if index[key] != r["sha256"]:
                raise ValueError("Existing feature identity changed: " + key)
            if key not in full or key not in cropped:
                raise ValueError("Incomplete existing features: " + key)
        else:
            pending.append(r)
    if not pending:
        print("All features are current; nothing changed.")
        return
    from scripts.predict import detector_model, feature_encoder, jpeg
    from src.models.localization import tractor_box, crop

    detector = detector_model()
    for r in pending:
        with Image.open(ROOT / r["path"]) as h:
            image = ImageOps.exif_transpose(h).convert("RGB")
        proposal = tractor_box(image, detector, 320, 544).get("foreground")
        view = jpeg(crop(image, proposal["box"]) if proposal else image)
        images = {"full": image, "crop": view}
        a = embeddings(feature_encoder("dinov2"), images, True)
        b = embeddings(feature_encoder("convnext"), images, True)
        full[r["key"]] = np.concatenate((a["full"], b["full"]))
        cropped[r["key"]] = np.concatenate((a["crop"], b["crop"]))
        index[r["key"]] = r["sha256"]
        print("Encoded", r["key"], flush=True)
    keys = ["full/" + k for k in sorted(full)] + ["crop/" + k for k in sorted(cropped)]
    values = [full[k] for k in sorted(full)] + [cropped[k] for k in sorted(cropped)]
    dest = ROOT / "datasets/features/recognition.npz"
    tmp = dest.with_name("recognition.pending.npz")
    np.savez_compressed(tmp, keys=np.array(keys), values=np.stack(values))
    tmp.replace(dest)
    (ROOT / "datasets/features/index.json").write_text(
        json.dumps(index, indent=2) + "\n", encoding="utf8"
    )


if __name__ == "__main__":
    main()
