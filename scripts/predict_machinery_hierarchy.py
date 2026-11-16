"""Inspect every detected machine with type/family/specialist heads and local OCR."""

import argparse
import json
from functools import lru_cache
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
from src.models.trailer_filter import PROMPT, split_trailers, remove_composite_views
from src.models.image_visibility import (assess_visibility, withhold_unreliable_details,
                                         withhold_ambiguous_surface_state)
from src.models.mvp_inference import (
    assess_conditions,
    load_optional_bundle,
    resolve_detected_type,
    verify_machine,
    verify_trailer,
)
from src.models.visible_condition import assess_visible_condition

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def detector_model():
    return load_detector(ROOT / "weights/grounding-dino-tiny")


@lru_cache(maxsize=2)
def feature_encoder(name):
    path = ROOT / "weights/dinov2-small.safetensors" if name == "dinov2" else CONVNEXT_WEIGHTS
    return encoder(name, path)


@lru_cache(maxsize=1)
def ocr_engine():
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1)


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


def attachment_semantics(image, detector):
    processor, model = detector
    inputs = processor(images=image, text=PROMPT,
                       size={"shortest_edge": 320, "longest_edge": 544}, return_tensors="pt")
    with torch.inference_mode():
        pred = model(**inputs)
    result = processor.post_process_grounded_object_detection(
        pred, inputs.input_ids, threshold=0.20, text_threshold=0.20,
        target_sizes=[(image.height, image.width)],
    )[0]
    return [dict(box=box, score=score, label=str(label))
            for box, score, label in zip(result["boxes"].tolist(), result["scores"].tolist(),
                                         result.get("text_labels", result.get("labels", [])), strict=True)]


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
    detector = detector_model()
    regions = detect(image, detector)
    semantic_ignored = []
    uncertain_regions = []
    semantic_proposals = []
    if len(regions) > 1:
        _, semantic_proposals = split_trailers(regions, attachment_semantics(image, detector))
    semantic_by_box = {tuple(r["bbox"]): r for r in semantic_proposals}
    legacy_view = None
    if regions:
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
        machine = legacy_view if len(regions) == 1 else jpeg(crop(image, region["box"]))
        context = image if len(regions) == 1 else machine
        files[f"{i}/full"] = context
        files[f"{i}/crop"] = machine
        if tuple(region["box"]) in semantic_by_box:
            # Classify the proposal itself; the 6% family-context padding may
            # contain a neighbouring tractor wheel and reverse its identity.
            files[f"{i}/attachment"] = jpeg(image.crop(region["box"]))
        images[i] = machine
    a = embeddings(feature_encoder("dinov2"), files, True)
    b = embeddings(feature_encoder("convnext"), files, True)
    vectors = {k: np.concatenate((a[k], b[k])) for k in files}
    kept = []
    ignored = []
    for i, region in enumerate(regions):
        is_machine, verification = verify_machine(
            vectors[f"{i}/crop"], box=region["box"], image_size=(image.width, image.height),
        )
        region["verification"] = verification
        trailer_proposal = semantic_by_box.get(tuple(region["box"]))
        if is_machine and (trailer_proposal or len(regions) == 1):
            is_trailer, reference_evidence = verify_trailer(vectors.get(f"{i}/attachment", vectors[f"{i}/crop"]))
            if is_trailer and trailer_proposal is None:
                _, proposals = split_trailers(
                    regions, attachment_semantics(image, detector_model()),
                    require_other_powered=False,
                )
                trailer_proposal = next((p for p in proposals if p["bbox"] == region["box"]), None)
            if is_trailer and trailer_proposal:
                rejected = dict(trailer_proposal, image_reference=reference_evidence)
                ignored.append(rejected)
                semantic_ignored.append(rejected)
                continue
        if is_machine and trailer_proposal and reference_evidence.get("uncertain_role"):
            folder = output.parent / "uncertain"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / (str(i) + ".jpg")
            image.crop(region["box"]).save(path, quality=95)
            uncertain = dict(bbox=region["box"], reason="ambiguous_powered_or_attachment",
                             crop=str(path.resolve()), image_reference=reference_evidence)
            uncertain_regions.append(uncertain)
            ignored.append(uncertain)
            continue
        if is_machine:
            kept.append((i, region))
        else:
            ignored.append(dict(
                bbox=region["box"],
                reason="attachment" if verification["label"] == "attachment" else "unconfirmed_region",
                verification=verification,
            ))
    kept, composite = remove_composite_views(kept, semantic_ignored)
    ignored.extend(composite)
    if (len(kept) == 1 and ignored
            and not (kept[0][1]["verification"] or {}).get("partial_rescue")):
        i, _ = kept[0]
        files = {f"{i}/full": image, f"{i}/crop": legacy_view}
        images[i] = legacy_view
        a = embeddings(feature_encoder("dinov2"), files, True)
        b = embeddings(feature_encoder("convnext"), files, True)
        vectors = {k: np.concatenate((a[k], b[k])) for k in files}
    engine = ocr_engine()
    front_path = ROOT / "output/tractor_parts/front_bundle.json"
    front_bundle = json.loads(front_path.read_bytes()) if front_path.is_file() else None
    objects = []
    for i, region in kept:
        machine = images[i]
        reading = inspect(machine, engine)
        vector = np.concatenate((vectors[f"{i}/full"], vectors[f"{i}/crop"]))
        resolved_type = resolve_detected_type(bundle, vector, region["type_candidate"])
        prediction = predict_hierarchy(bundle, vector, reading, resolved_type)
        prediction["detector_type_candidate"] = region["type_candidate"]
        prediction["region_verification"] = region["verification"]
        if len(kept) > 1:
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
        if semantic_ignored:
            prediction.update(
                status="manual_review", family=None, family_name=None,
                calibration_domain="semantic_trailer_removal_not_validated",
                reason="Прицеп исключён из числа машин; семейство трактора требует проверки человеком.",
            )
        if uncertain_regions:
            prediction.update(status="manual_review", family=None, family_name=None,
                              calibration_domain="ambiguous_region_role",
                              reason="Есть неподтверждённая область: возможно, это орудие или фрагмент машины. Нужна проверка человеком.")
        folder = output.parent / "objects" / str(i)
        folder.mkdir(parents=True, exist_ok=True)
        machine.save(folder / "machine.jpg", quality=95)
        if front_bundle and prediction["family_candidate"] == "mtz_80_family":
            side = linear(front_bundle["orientation"], vectors[f"{i}/crop"])
            front_region(machine, side).save(folder / "front_proposal.jpg", quality=95)
            prediction["details"]["front_axle"] = dict(
                status="not_verified",
                orientation_proposal=side,
                image=str((folder / "front_proposal.jpg").resolve()),
                warning="Region proposal only; does not establish driven/undriven axle or model.",
            )
        attributes = assess_conditions(
            vectors[f"{i}/full"], vectors[f"{i}/crop"],
            image=image if len(kept) == 1 else machine,
        )
        if (attributes["snow"].get("value") is True
                and not prediction.get("visible_model_marking")):
            prediction.update(
                status="manual_review", family=None, family_name=None,
                reason="В кадре снег; семейство требует проверки по маркировке и деталям машины.",
            )
        withhold_ambiguous_surface_state(
            attributes, region["box"], [r["bbox"] for r in uncertain_regions],
            contextual=len(kept) == 1,
        )
        visibility = assess_visibility(image, machine)
        withhold_unreliable_details(
            prediction, attributes, visibility,
            partial=bool(region["verification"] and region["verification"].get("partial_rescue")),
        )
        dirt = attributes["dirty"]
        condition = "not_assessed"
        condition_reason = dirt["text"]
        if dirt["status"] == "assessed":
            condition = "dirty" if dirt["value"] else "clean"
            condition_reason = None
        elif load_optional_bundle("condition_bundle") is None:
            condition, condition_reason = assess_visible_condition(
                image if len(kept) == 1 else machine,
                prediction["family_candidate"], single_machine=True
            )
        prediction.update(
            object_id=len(objects),
            detector_region_id=i,
            bbox=region["box"],
            marking_observations=reading,
            crop=str((folder / "machine.jpg").resolve()),
            condition=condition,
            condition_reason=condition_reason,
            condition_attributes=attributes,
        )
        objects.append(prediction)
    result = dict(
        experimental=True,
        image=str(image_path.resolve()),
        sha256=digest(image_path),
        bundle_sha256=digest(bundle_path),
        objects=objects,
        ignored_regions=ignored,
        uncertain_regions=uncertain_regions,
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
