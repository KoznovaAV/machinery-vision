"""Visibility and attribution limits; they do not infer make, weather, or dirt."""

import numpy as np
from PIL import ImageOps


def assess_visibility(scene, machine):
    def median(image):
        gray = ImageOps.grayscale(image)
        gray.thumbnail((224, 224))
        return float(np.median(np.asarray(gray, dtype=np.float32)) / 255)

    scene_median = median(scene)
    machine_median = median(machine)
    return dict(
        low_light=scene_median < 0.15 and machine_median < 0.25,
        scene_median_luminance=scene_median,
        machine_median_luminance=machine_median,
        policy="prototype_photometric_refusal_v1",
    )


def withhold_unreliable_details(prediction, attributes, visibility, partial=False):
    prediction["image_quality"] = visibility
    if not visibility["low_light"] and not partial:
        return
    reason = (
        "Недостаточное освещение: модель и состояние не удалось надёжно оценить. "
        "Нужен снимок при рассеянном свете без блика в камеру."
        if visibility["low_light"]
        else "Машина видна частично: для определения семейства и состояния нужен полный кадр."
    )
    prediction.update(
        status="manual_review",
        family=None,
        family_name=None,
        raw_family_candidate=prediction.get("family_candidate"),
        family_candidate=None,
        family_candidate_name=None,
        label_candidate=None,
        model_candidate=None,
        model_candidate_name=None,
        reason=reason,
        calibration_domain=(
            "low_light_not_validated" if visibility["low_light"] else "partial_object_not_validated"
        ),
    )
    for attribute in attributes:
        attributes[attribute] = dict(status="not_assessed", value=None, text=reason)


def withhold_ambiguous_surface_state(attributes, box, uncertain_boxes, contextual=False):
    """Do not attribute dirt/water to a machine when a disputed region overlaps it.

    Scene weather is kept separate from the state of a particular machine. A
    contextual single-object view may also contain an uncertain attached object.
    """
    overlaps = any(
        max(0, min(box[2], b[2]) - max(box[0], b[0]))
        * max(0, min(box[3], b[3]) - max(box[1], b[1]))
        > 0
        for b in uncertain_boxes
    )
    if not overlaps and not (contextual and uncertain_boxes):
        return False
    reason = (
        "Не удалось однозначно отделить поверхность машины от спорного оборудования. "
        "Для оценки загрязнения и влажности нужен отдельный кадр машины."
    )
    for name in ("dirty", "wet"):
        if name in attributes:
            attributes[name] = dict(status="not_assessed", value=None, text=reason)
    return True
