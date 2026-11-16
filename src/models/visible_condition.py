"""Conservative estimate of visible dirt for a supported tractor on a single photo."""

import logging
from functools import lru_cache

from PIL import Image

from src.config.classes import class_to_idx, state_to_idx
from src.config.config_loader import load_config
from src.config.hierarchy_artifacts import ROOT
from src.data.transforms import get_val_transforms
from src.models.loader import load_multi_task_model
from src.models.predict import predict_image

LOGGER = logging.getLogger(__name__)
MTZ_FAMILIES = {"mtz_80_family", "mtz_1221_family", "mtz_1523_family"}
CLEAN_MAX_DIRT_SCORE = 0.20
DIRTY_MIN_DIRT_SCORE = 0.80


@lru_cache(maxsize=1)
def _load_estimator():
    config = load_config()
    entry = config.models.get("machine")
    if entry is None or "state" not in entry.tasks:
        raise ValueError("No checkpoint configured for the condition estimate")
    return load_multi_task_model(ROOT / entry.checkpoint), get_val_transforms(config.image_size)


def assess_visible_condition(
    image: Image.Image, family: str | None, *, single_machine: bool
) -> tuple[str, str | None]:
    """Return the estimated cleanliness and an abstention reason, when applicable."""
    if not single_machine:
        return "not_assessed", "Для оценки нужен отдельный кадр каждой машины."
    if family not in MTZ_FAMILIES:
        return "not_assessed", "Оценку загрязнения пока применяем к распознанным тракторам МТЗ."
    try:
        model, transform = _load_estimator()
        kind, _, state, confidence = predict_image(model, image, transform)
    except (OSError, RuntimeError, ValueError):
        LOGGER.warning("Optional condition estimator unavailable", exc_info=True)
        return "not_assessed", "Модуль оценки загрязнения недоступен."
    if kind != class_to_idx("mtz_belarus"):
        return "not_assessed", "Классификаторы разошлись в определении семейства."

    dirty_score = confidence if state == state_to_idx("dirty") else 1.0 - confidence
    if dirty_score <= CLEAN_MAX_DIRT_SCORE:
        return "clean", None
    if dirty_score >= DIRTY_MIN_DIRT_SCORE:
        return "dirty", None
    return "not_assessed", "На фото недостаточно уверенных признаков загрязнения."
