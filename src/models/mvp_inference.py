"""Optional reviewed MVP heads for region verification and visible conditions."""

import json
import hashlib
from functools import lru_cache

import numpy as np

from src.config.hierarchy_artifacts import ROOT
from src.config.machinery_taxonomy import CATALOG
from src.models.condition_features import weather_measurements
from src.models.hierarchy import qualified, vote


@lru_cache(maxsize=3)
def load_optional_bundle(name):
    path = ROOT / "artifacts/mvp" / (name + ".json")
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def verify_machine(vector, *, box=None, image_size=None):
    bundle = load_optional_bundle("region_bundle")
    if bundle is None:
        return True, None
    label, score, _ = vote(bundle["head"], vector)
    minimum = bundle.get("minimum_machine_score", 0.80)
    accepted = label == "machine" and score >= minimum
    evidence = dict(label=label, score=score, minimum_machine_score=minimum,
                    partial_rescue=False)
    if not accepted and box is not None and image_size is not None and bundle.get("partial_head"):
        width, height = image_size
        touches_edge = box[0] <= 1 or box[1] <= 1 or box[2] >= width - 1 or box[3] >= height - 1
        if touches_edge:
            fragment, fragment_score, _ = vote(bundle["partial_head"], vector)
            minimum_partial = bundle.get("partial_minimum_score", 0.95)
            accepted = (fragment == "machine" and fragment_score >= minimum_partial
                        and qualified(bundle.get("partial_calibration", {}), fragment, fragment_score))
            evidence.update(partial_label=fragment, partial_score=fragment_score,
                            partial_rescue=bool(accepted))
    return accepted, evidence


@lru_cache(maxsize=2)
def _trailer_references(path, expected_sha256):
    target = (ROOT / path).resolve()
    if not target.is_relative_to(ROOT.resolve()):
        raise ValueError("Reference path outside project")
    if hashlib.sha256(target.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("Changed trailer reference vectors")
    with np.load(target, allow_pickle=False) as data:
        return data["vectors"].copy(), data["labels"].copy(), data["keys"].copy()


def verify_trailer(vector, *, bundle=None):
    """Require calibrated image-reference evidence in addition to semantic alignment.

    The margin is a cosine-similarity difference, never a probability. Missing or
    changed reference data cannot remove a machine. This verifier is not used alone.
    """
    bundle = load_optional_bundle("region_bundle") if bundle is None else bundle
    config = (bundle or {}).get("trailer_reference")
    if not config or config.get("threshold") is None:
        return False, dict(status="unavailable")
    try:
        references, labels, keys = _trailer_references(config["path"], config["sha256"])
        value = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(value))
        if norm <= 0 or not np.isfinite(value).all():
            raise ValueError("Invalid reference vector")
        query = value / norm
        if config.get("normalization") == "per_encoder_l2":
            if query.shape != (1152,):
                raise ValueError("Unexpected encoder dimensions")
            blocks = (query[:384], query[384:])
            query = np.concatenate([v / max(float(np.linalg.norm(v)), 1e-9) for v in blocks])
            query /= max(float(np.linalg.norm(query)), 1e-9)
        scores = references @ query
        attachment_indices = np.flatnonzero(labels == "attachment")
        machine_indices = np.flatnonzero(labels == "machine")
        negative = attachment_indices[np.argmax(scores[attachment_indices])]
        positive = machine_indices[np.argmax(scores[machine_indices])]
        margin = float(scores[negative] - scores[positive])
        minimum = config["threshold"]
        accepted = margin >= minimum
        evidence = dict(
            status="assessed", score_kind="cosine_margin", margin=margin,
            threshold=minimum, attachment_reference=str(keys[negative]),
            machine_reference=str(keys[positive]), uncertain_role=False,
        )
        if config.get("supplement_head"):
            label, score, _ = vote(config["supplement_head"], value / norm * config.get("supplement_scale", 20))
            calibration = config.get("supplement_calibration", {})
            attachment = label == "attachment" and qualified(calibration, label, score)
            machine = label == "machine" and qualified(calibration, label, score)
            accepted = accepted or (margin >= config.get("supplement_minimum_margin", 0) and attachment)
            calibrated = all(calibration.get(k, {}).get("threshold") is not None for k in ("attachment", "machine"))
            uncertain = (calibrated and not accepted and not machine
                         and abs(margin) < config.get("minimum_margin", .03))
            evidence.update(supplement_label=label, supplement_score=score,
                            uncertain_role=bool(uncertain))
        return bool(accepted), evidence
    except (OSError, ValueError, KeyError, IndexError):
        return False, dict(status="unavailable")


def resolve_detected_type(bundle, vector, detected_type):
    """Resolve detector category noise only when both trained heads agree."""
    vehicle, _, _ = vote(bundle["heads"]["type"], vector)
    family, _, _ = vote(bundle["heads"]["family"], vector)
    family_type = CATALOG["families"].get(family, {}).get("type")
    if detected_type in {None, "tractor", "combine_harvester"} and vehicle == family_type:
        return vehicle
    return detected_type


def assess_conditions(full_vector, crop_vector, image=None, *, bundle=None):
    if bundle is None:
        bundle = load_optional_bundle("condition_bundle")
    result = {}
    names = {
        "dirty": ("Видимой грязи не обнаружено", "Есть признаки загрязнения"),
        "snow": ("Снег в кадре не подтверждён", "В зоне машины виден снег"),
        "rain": ("Видимого дождя нет", "В кадре виден дождь"),
        "wet": ("Влажность поверхности не подтверждена", "Есть признаки мокрой поверхности"),
    }
    for attribute in names:
        if bundle is None or attribute not in bundle["heads"]:
            result[attribute] = dict(
                status="not_assessed",
                text="Недостаточно проверенных данных для оценки.",
            )
            continue
        source = bundle.get("feature_sources", {}).get(attribute)
        vector = full_vector if source == "scene" or attribute == "rain" else crop_vector
        measurements = None
        if source == "state":
            from src.models.predict import predict_image
            from src.models.visible_condition import _load_estimator

            try:
                if image is None:
                    raise ValueError("Missing source photo")
                model, transform = _load_estimator()
                _, _, state, confidence = predict_image(model, image, transform)
                probability = confidence if state == 1 else 1.0 - confidence
                probability = np.clip(probability, 1e-5, 1 - 1e-5)
                vector = np.array([np.log(probability / (1 - probability))])
            except (OSError, RuntimeError, ValueError):
                result[attribute] = dict(status="not_assessed", value=None,
                                         text="Модуль оценки загрязнения недоступен.")
                continue
        if source in {"weather_pixels", "scene_weather"}:
            if image is None:
                result[attribute] = dict(status="not_assessed", value=None,
                                         text="Для оценки нужен исходный кадр.")
                continue
            measurements = weather_measurements(image)
            vector = measurements * 5
            if source == "scene_weather":
                vector = np.concatenate((full_vector, vector))
        label, score, _ = vote(bundle["heads"][attribute], vector)
        floor = bundle.get("minimum_scores", {}).get(attribute, {}).get(label, 0.0)
        accepted = qualified(bundle["calibration"][attribute], label, score) and score >= floor
        if attribute == "snow" and bundle.get("snow_support_required"):
            visible_support = (measurements is not None and measurements[5] >= 0.15
                               and measurements[6] >= 0.50)
            accepted = accepted and label == "1" and visible_support
        result[attribute] = dict(
            status="assessed" if accepted else "not_assessed",
            value=bool(int(label)) if accepted else None,
            text=names[attribute][int(label)] if accepted else "Недостаточно уверенных признаков.",
            score=score,
        )
    if (result["snow"].get("value") is True
            and bundle is not None and bundle.get("withhold_dirt_in_snow")):
        result["dirty"] = dict(
            status="not_assessed", value=None,
            text="Снег может скрывать загрязнение; нужен более крупный план поверхности.",
        )
    return result
