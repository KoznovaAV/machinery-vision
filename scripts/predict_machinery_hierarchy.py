"""Inspect every detected machine with type/family/specialist heads and local OCR."""

import argparse
import json
from io import BytesIO
from pathlib import Path

import albumentations  # noqa: F401
import numpy as np
import torch
from PIL import Image, ImageOps

from scripts.analyze_model_marking import inspect
from scripts.benchmark_front_axle import front_region, linear
from scripts.benchmark_grouped_recognition import digest, embeddings, encoder
from scripts.benchmark_tractor_localization import crop, load_detector, tractor_box
from scripts.predict_tractor_conditions import CONVNEXT_WEIGHTS
from src.config.hierarchy_artifacts import hierarchy_bundle_path
from src.models.hierarchy import predict_hierarchy
from src.models.machinery_regions import select_regions

ROOT = Path(__file__).resolve().parents[1]


def jpeg(image):
    b = BytesIO()
    image.save(b, format="JPEG", quality=95)
    b.seek(0)
    with Image.open(b) as im:
        return im.convert("RGB")


def detect(image, detector):
    processor, model = detector
    inputs = processor(
        images=image,
        text="tractor. combine harvester. bulldozer.",
        size={"shortest_edge": 320, "longest_edge": 544},
        return_tensors="pt",
    )
    with torch.inference_mode():
        pred = model(**inputs)
    result = processor.post_process_grounded_object_detection(
        pred,
        inputs.input_ids,
        threshold=0.25,
        text_threshold=0.2,
        target_sizes=[(image.height, image.width)],
    )[0]
    labels = result.get("text_labels", result.get("labels", []))
    raw = [
        dict(box=box, score=score, label=str(label))
        for box, score, label in zip(
            result["boxes"].tolist(), result["scores"].tolist(), labels, strict=True
        )
    ]
    return select_regions(raw, image.width, image.height)


def run(image_path, bundle_path, output):
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    primary = ROOT / "weights/dinov2-small.safetensors"
    detector_path = ROOT / "weights/grounding-dino-tiny"
    for field, path in [
        ("encoder_sha256", primary),
        ("secondary_encoder_sha256", CONVNEXT_WEIGHTS),
        ("detector_sha256", detector_path / "model.safetensors"),
    ]:
        if bundle[field] != digest(path):
            raise ValueError("Changed hierarchy component " + field)
    if bundle["detector_size"] != [320, 544] or bundle["input"] != 224:
        raise ValueError("Changed hierarchy preprocessing")
    torch.set_num_threads(5)
    with Image.open(image_path) as handle:
        image = ImageOps.exif_transpose(handle).convert("RGB")
    detector = load_detector(detector_path)
    regions = detect(image, detector)
    legacy_view = None
    if len(regions) == 1:
        prior = tractor_box(image, detector, 320, 544).get("foreground")
        legacy_view = jpeg(crop(image, prior["box"]) if prior else image)
    del detector
    output.parent.mkdir(parents=True, exist_ok=True)
    if not regions:
        result = dict(
            experimental=True,
            status="no_supported_object_detected",
            objects=[],
            model=None,
            exact_model=None,
        )
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    files = {}
    images = {}
    for i, region in enumerate(regions):
        machine = legacy_view if legacy_view is not None else jpeg(crop(image, region["box"]))
        context = image if len(regions) == 1 else machine
        files[f"{i}/full"] = context
        files[f"{i}/crop"] = machine
        images[i] = machine
    a = embeddings(encoder("dinov2", primary), files, True)
    b = embeddings(encoder("convnext", CONVNEXT_WEIGHTS), files, True)
    vectors = {k: np.concatenate((a[k], b[k])) for k in files}
    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1)
    front_bundle = json.loads(
        (ROOT / "output/tractor_parts/front_bundle.json").read_text(encoding="utf-8")
    )
    objects = []
    for i, region in enumerate(regions):
        machine = images[i]
        reading = inspect(machine, engine)
        vector = np.concatenate((vectors[f"{i}/full"], vectors[f"{i}/crop"]))
        prediction = predict_hierarchy(bundle, vector, reading, region["type_candidate"])
        if len(regions) > 1:
            prediction.update(
                status="manual_review",
                family=None,
                family_name=None,
                reason=(
                    "Несколько машин обработаны отдельно; "
                    "калибровка этого режима ещё не подтверждена."
                ),
                calibration_domain="multi_object_not_validated",
            )
        else:
            prediction["calibration_domain"] = "legacy_single_object_features"
        folder = output.parent / "objects" / str(i)
        folder.mkdir(parents=True, exist_ok=True)
        machine.save(folder / "machine.jpg", quality=95)
        if prediction["family_candidate"] == "mtz_80_family":
            side = linear(front_bundle["orientation"], vectors[f"{i}/crop"])
            front_region(machine, side).save(folder / "front_proposal.jpg", quality=95)
            prediction["details"]["front_axle"] = dict(
                status="not_verified",
                orientation_proposal=side,
                image=str((folder / "front_proposal.jpg").resolve()),
                warning="Region proposal only; does not establish driven/undriven axle or model.",
            )
        prediction.update(
            object_id=i,
            bbox=region["box"],
            marking_observations=reading,
            crop=str((folder / "machine.jpg").resolve()),
            condition="not_assessed",
        )
        objects.append(prediction)
    result = dict(
        experimental=True,
        image=str(image_path.resolve()),
        sha256=digest(image_path),
        bundle_sha256=digest(bundle_path),
        objects=objects,
        warning=(
            "Separate experimental hierarchy; "
            "exact identities require further validated evidence."
        ),
    )
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--bundle", type=Path, default=hierarchy_bundle_path()
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    r = run(args.image, args.bundle, args.output)
    print(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
