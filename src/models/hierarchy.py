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


def predict_hierarchy(bundle, vector, marking=None, detected_type=None):
    if (
        bundle.get("taxonomy_version") != TAXONOMY_VERSION
        or bundle.get("taxonomy_sha256") != TAXONOMY_SHA256
        or bundle.get("decision_policy") != "hierarchical_review_only"
    ):
        raise ValueError("Unsupported hierarchy bundle")
    vehicle, vehicle_score, _ = vote(bundle["heads"]["type"], vector)
    family, score, _ = vote(bundle["heads"]["family"], vector)
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
    if marking_label and not conflict:
        candidate = marking_label
    expected_type = next((v["type"] for v in LABELS.values() if v["family"] == family), None)
    type_conflict = bool(expected_type and vehicle != expected_type) or bool(
        detected_type and detected_type != vehicle
    )
    known = (
        family != "unknown"
        and qualified(bundle["calibration"]["family"], family, score)
        and not conflict
        and not type_conflict
    )
    type_known = (
        qualified(bundle["calibration"]["type"], vehicle, vehicle_score) and not type_conflict
    )
    resolution = LABELS.get(candidate, {}).get("resolution", "unknown")
    visible = visible_marking(marking) if marking_label and not conflict else None
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
    return dict(
        schema_version="hierarchical-response-v1",
        experimental=True,
        status="family_only" if known else "manual_review",
        vehicle_type=vehicle if type_known else None,
        vehicle_type_candidate=detected_type or vehicle,
        family=family if known else None,
        family_candidate=family,
        family_name=FAMILY_NAMES.get(family) if known else None,
        family_candidate_name=FAMILY_NAMES.get(family),
        model=None,
        exact_model=None,
        label_candidate=candidate if candidate != "other" and not type_conflict else None,
        model_candidate=fine_candidate if not type_conflict else None,
        candidate_resolution=resolution,
        alternatives=alternatives,
        marking_candidate=marking_label,
        visible_model_marking=visible,
        marking_conflict=conflict,
        type_conflict=type_conflict,
        reason=(
            "Семейство определено; точная модель не подтверждена."
            if known
            else "Недостаточно согласованных подтверждений семейства и точной модели."
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
