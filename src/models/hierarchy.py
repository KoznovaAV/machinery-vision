"""Conservative hierarchical candidate inference; never certify an exact model."""

from __future__ import annotations

import re

import numpy as np

from src.config.machinery_taxonomy import (
    CATALOG,
    FAMILY_NAMES,
    LABELS,
    TAXONOMY_SHA256,
    TAXONOMY_VERSION,
    family_of,
    required_details,
)


def logits(head, vector):
    coef = np.asarray(head["coef"])
    intercept = np.asarray(head["intercept"])
    vector = np.asarray(vector)
    if coef.ndim != 2 or coef.shape[1] != len(vector):
        raise ValueError("Incompatible feature dimensions")
    raw = coef @ vector + intercept
    if len(head["classes"]) == 2 and raw.shape == (1,):
        raw = np.array([-raw[0] / 2, raw[0] / 2])
    if raw.shape != (len(head["classes"]),) or not np.isfinite(raw).all():
        raise ValueError("Invalid head logits")
    return raw


def vote(head, vector):
    raw = logits(head, vector)
    p = np.exp(raw - raw.max())
    p /= p.sum()
    order = np.argsort(p)[::-1]
    return (
        str(head["classes"][int(order[0])]),
        float(p[order[0]]),
        [str(head["classes"][int(i)]) for i in order[:3]],
    )


def qualified(calibration, label, score):
    threshold = calibration.get(label, {}).get("threshold")
    return threshold is not None and score >= threshold


def visible_marking(evidence):
    tokens = set()
    for pair in (evidence or {}).get("pairs", []):

        def read(text):
            text = text.upper().replace("Т", "T").replace(",", ".")
            return {
                re.sub(r"\s+", "", token)
                for token in re.findall(
                    r"(?<![0-9])(?:T\s*50[3-7]0|820(?:\.1)?|82(?:\.1)?|"
                    r"80(?:\.1)?|1523(?:\.\d+)?)(?![0-9])",
                    text,
                )
            }

        tokens.update(read(pair["full_text"]) & read(pair["body_text"]))
    return next(iter(tokens)) if len(tokens) == 1 else None


def branded_marking_family(marking):
    """Offer a family hint only for model digits corroborated near a readable brand."""
    if not marking or marking.get("status") != "consistent_marking":
        return None
    label = marking.get("suggestion")
    token = visible_marking(marking)
    size = marking.get("image_size", [])
    if label not in LABELS or token is None or len(size) != 2 or min(size) <= 0:
        return None
    family = family_of(label)
    patterns = {
        "mtz_80_family": r"(?<![A-ZА-Я])(?:MTZ|МТЗ|BELARUS|БЕЛАРУС)(?![A-ZА-Я])",
        "nh_t5000_family": r"NEW\s*HOLLAND",
    }
    if family == "mtz_80_family" and not re.fullmatch(r"(?:80|82|820)(?:\.1)?", token):
        return None
    if family == "nh_t5000_family" and not re.fullmatch(r"T50[3-7]0", token):
        return None
    pattern = patterns.get(family)
    if pattern is None:
        return None
    width, height = size
    lines = marking.get("full", []) + marking.get("body", [])
    for pair in marking.get("pairs", []):
        box = pair.get("full_box")
        if not box or len(box) != 4:
            continue
        x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        if not 0.20 <= y / height <= 0.75:
            continue
        for line in lines:
            if line.get("confidence", 0) < 0.95 or not re.search(
                pattern, line.get("text", "").upper()
            ):
                continue
            brand = line.get("box", [])
            if len(brand) != 4:
                continue
            dx = ((brand[0] + brand[2]) / 2 - x) / width
            dy = ((brand[1] + brand[3]) / 2 - y) / height
            if dx * dx + dy * dy <= 0.20**2:
                return family
    return None


def unsupported_nh_marking(marking):
    """Recognize corroborated NH M/TM/TS markings outside the trained catalogue."""
    if not marking or marking.get("status") == "consistent_marking":
        return None
    size = marking.get("image_size", [])
    if len(size) != 2 or min(size) <= 0:
        return None
    width, height = size
    pattern = r"(?<![A-Z0-9])(?:TM|TS|M)[- ]?\d{2,3}(?![A-Z0-9])"

    def tokens(line):
        if line.get("confidence", 0) < 0.95:
            return set()
        text = line.get("text", "").upper().translate(str.maketrans({"М": "M", "Т": "T"}))
        return {re.sub(r"[- ]", "", token) for token in re.findall(pattern, text)}

    def overlaps(a, b):
        if len(a) != 4 or len(b) != 4:
            return False
        intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
            0, min(a[3], b[3]) - max(a[1], b[1])
        )
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
        return union > 0 and intersection / union >= 0.05

    full, body = marking.get("full", []), marking.get("body", [])
    brands = []
    for a in full:
        for b in body:
            if (
                min(a.get("confidence", 0), b.get("confidence", 0)) >= 0.95
                and all(
                    re.search(r"NEW\s*HOLLAND", line.get("text", "").upper()) for line in (a, b)
                )
                and overlaps(a.get("box", []), b.get("box", []))
            ):
                brands.append(a["box"])
    candidates = set()
    for a in full:
        for b in body:
            common = tokens(a) & tokens(b)
            if not common or not overlaps(a.get("box", []), b.get("box", [])):
                continue
            box = a["box"]
            x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            if not 0.20 <= y / height <= 0.75:
                continue
            for brand in brands:
                dx = ((brand[0] + brand[2]) / 2 - x) / width
                dy = ((brand[1] + brand[3]) / 2 - y) / height
                if dx * dx + dy * dy <= 0.20**2:
                    candidates.update(common)
    trained = {item.get("evidence_token") for item in CATALOG["models"].values()}
    candidates -= trained
    return next(iter(candidates)) if len(candidates) == 1 else None


def predict_hierarchy(bundle, vector, marking=None, detected_type=None):
    if (
        bundle.get("taxonomy_version") != TAXONOMY_VERSION
        or bundle.get("taxonomy_sha256") != TAXONOMY_SHA256
        or bundle.get("decision_policy") != "hierarchical_review_only"
    ):
        raise ValueError("Unsupported hierarchy bundle")
    vehicle, vehicle_score, _ = vote(bundle["heads"]["type"], vector)
    family, score, _ = vote(bundle["heads"]["family"], vector)
    family_calibration = bundle["calibration"]["family"]
    nh_specialist = bundle["heads"].get("nh_specialist")
    if nh_specialist is not None:
        nh_label, nh_score, _ = vote(nh_specialist, vector)
        if nh_label == "nh_t5000_family" and qualified(
            bundle["calibration"].get("nh_specialist", {}), nh_label, nh_score
        ):
            family, score = nh_label, nh_score
            family_calibration = bundle["calibration"]["nh_specialist"]
    unknown_gate = bundle["heads"].get("unknown_gate")
    if unknown_gate is not None:
        novelty, novelty_score, _ = vote(unknown_gate, vector)
        if novelty == "unknown" and qualified(
            bundle["calibration"].get("unknown_gate", {}), novelty, novelty_score
        ):
            family, score = "unknown", novelty_score
    classifier_family = family
    unsupported_marking = unsupported_nh_marking(marking)
    if unsupported_marking:
        family = "unknown"
    eligible = [k for k, v in LABELS.items() if v["family"] == family]
    if family == "mtz_80_family":
        candidate, _, alternatives = vote(bundle["heads"]["mtz_specialist"], vector)
    else:
        candidate = eligible[0] if len(eligible) == 1 else "other"
        alternatives = eligible[:3]
    marking_label = (
        marking.get("suggestion")
        if marking and marking.get("status") == "consistent_marking"
        else None
    )
    conflict = bool(
        marking_label and (marking_label not in LABELS or family_of(marking_label) != family)
    )
    marking_hint = branded_marking_family(marking) if conflict else None
    if marking_hint is not None:
        family = marking_hint
        candidate = marking_label
        alternatives = [k for k, value in LABELS.items() if value["family"] == family]
    elif marking_label and not conflict:
        candidate = marking_label
    expected_type = next((v["type"] for v in LABELS.values() if v["family"] == family), None)
    type_conflict = bool(expected_type and vehicle != expected_type) or bool(
        detected_type and detected_type != vehicle
    )
    known = (
        family != "unknown"
        and qualified(family_calibration, family, score)
        and not conflict
        and not type_conflict
    )
    type_known = (
        qualified(bundle["calibration"]["type"], vehicle, vehicle_score) and not type_conflict
    )
    resolution = LABELS.get(candidate, {}).get("resolution", "unknown")
    visible = unsupported_marking or (visible_marking(marking) if marking_label else None)
    exact_candidates = [
        k
        for k, v in CATALOG["models"].items()
        if v["family"] == family and v["evidence_token"] == visible
    ]
    fine_candidate = (
        exact_candidates[0]
        if len(exact_candidates) == 1
        else candidate if resolution == "model" else None
    )
    model_candidate_name = CATALOG["models"].get(fine_candidate, {}).get("name")
    if visible and family in {"mtz_1221_family", "mtz_1523_family"}:
        prefix = "1221" if family == "mtz_1221_family" else "1523"
        if re.fullmatch(re.escape(prefix) + r"\.\d+", visible):
            model_candidate_name = "БЕЛАРУС " + visible
    return dict(
        schema_version="hierarchical-response-v1",
        experimental=True,
        status="family_only" if known else "manual_review",
        vehicle_type=vehicle if type_known else None,
        vehicle_type_candidate=detected_type or vehicle,
        family=family if known else None,
        family_candidate=family,
        classifier_family_candidate=classifier_family,
        candidate_source=(
            "unsupported_marking"
            if unsupported_marking
            else "branded_marking" if marking_hint else "visual_classifiers"
        ),
        unsupported_model_marking=unsupported_marking,
        family_name=FAMILY_NAMES.get(family) if known else None,
        family_candidate_name=FAMILY_NAMES.get(family),
        model=None,
        exact_model=None,
        label_candidate=candidate if candidate != "other" and not type_conflict else None,
        model_candidate=fine_candidate if not type_conflict else None,
        model_candidate_name=model_candidate_name if not type_conflict else None,
        candidate_resolution=resolution,
        alternatives=alternatives,
        marking_candidate=marking_label,
        visible_model_marking=visible,
        marking_conflict=conflict,
        type_conflict=type_conflict,
        reason=(
            f"Видна маркировка New Holland {unsupported_marking}; эта серия отсутствует в обученном справочнике."
            if unsupported_marking
            else (
                "Семейство определено; точная модель не подтверждена."
                if known
                else (
                    "Маркировка и классификатор расходятся; кандидат указан по маркировке, нужна проверка человеком."
                    if marking_hint
                    else "Недостаточно согласованных подтверждений семейства и точной модели."
                )
            )
        ),
        requested_photos=required_details(family),
        details=dict(
            front_axle="not_verified",
            marking="corroborated_text" if marking_label else "not_confirmed",
        ),
        warning=(
            "OOF thresholds are experimental; "
            "model groups and series are never exact identities."
        ),
    )
