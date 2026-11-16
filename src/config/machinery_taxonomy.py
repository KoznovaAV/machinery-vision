"""Single source of truth for experimental machinery label granularity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

CATALOG_PATH = Path(__file__).resolve().parents[2] / "configs/machinery_taxonomy.json"
_RAW = CATALOG_PATH.read_bytes()
CATALOG = json.loads(_RAW.decode("utf-8"))
TAXONOMY_VERSION = CATALOG["version"]
TAXONOMY_SHA256 = hashlib.sha256(_RAW).hexdigest()
LABELS = CATALOG["labels"]
MTZ_SPECIALIST = frozenset(CATALOG["specialists"]["mtz_80_family"])
FAMILY_NAMES = {k: v["name"] for k, v in CATALOG["families"].items()}
TYPE_NAMES = CATALOG["types"]
for _label, _spec in LABELS.items():
    if (
        _spec["family"] not in FAMILY_NAMES
        or CATALOG["families"][_spec["family"]]["type"] != _spec["type"]
    ):
        raise ValueError("Inconsistent family/type in machinery taxonomy: " + _label)
for _model, _spec in CATALOG["models"].items():
    _parent = LABELS.get(_spec["legacy_label"])
    if not _parent or any(_parent[k] != _spec[k] for k in ("type", "family")):
        raise ValueError("Inconsistent model parent in machinery taxonomy: " + _model)


def family_of(label):
    return LABELS[label]["family"] if label in LABELS else "unknown"


def type_of(label):
    return LABELS[label]["type"] if label in LABELS else "unknown"


def label_resolution(label):
    return LABELS[label]["resolution"] if label in LABELS else "unknown"


def required_details(family):
    if family == "mtz_80_family":
        return ["передний мост крупным планом", "маркировка на капоте"]
    if family in {"nh_t5000_family", "claas_axion_800_family"}:
        return ["маркировка точной модели на капоте"]
    return ["маркировка модели", "общий вид машины сбоку"]
