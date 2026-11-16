"""Marking token boundaries and conservative part-view proposals."""

import re
import numpy as np


def marking_labels(text):
    """Conservative token parsing: 820 is not 82, and 80.1 is not 80."""
    text = text.upper().replace("Т", "T").replace("О", "O")
    result = set()
    for token in re.findall(r"(?<![A-Z0-9])(?:T\s*)?\d{2,4}(?:[.,]\d+)?(?![A-Z0-9])", text):
        token = token.replace(" ", "").replace(",", ".")
        if re.fullmatch(r"T50[3-7]0", token):
            result.add("nh_t5000")
        elif token in {"80", "80.1"}:
            result.add("mtz_80")
        elif token in {"82", "82.1"}:
            result.add("mtz_82")
        elif token in {"820", "820.1"}:
            result.add("mtz_820")
    if re.search(r"\b(?:TD\s*\d+[A-Z]*|T[4-9]\s*\.\s*\d+[A-Z]*|ARION)\b", text):
        result.add("outside_supported")
    return sorted(result)


def lower_proposal(image):
    # Includes both ends: a fixed window cannot know which side is the front.
    return image.crop((0, int(image.height * 0.48), image.width, image.height))


def marking_candidates(lines):
    """Prefer explicit model codes over isolated numbers on implements or plates.

    This is a review/veto hint, never permission to identify a tractor exactly.
    """
    reliable = [line for line in lines if line["confidence"] >= 0.9]
    explicit = []
    for line in reliable:
        text = line["text"].upper().replace("Т", "T")
        if re.search(r"T\s*50[3-7]0|ARION|\bTD\s*\d|(?:MT3|MTZ|МТЗ|BELARUS)\s*[- ]?\s*\d", text):
            explicit.extend(line["candidates"])
    if explicit:
        return sorted(set(explicit))
    return sorted(
        {
            candidate
            for line in reliable
            if not re.search(r"\d\s*[:;]", line["text"])
            for candidate in line["candidates"]
        }
    )


def front_region(image, side):
    w, h = image.size
    boxes = {
        "L": (0, 0.45, 0.62, 1),
        "R": (0.38, 0.45, 1, 1),
        "center": (0.12, 0.45, 0.88, 1),
        "rear": (0, 0.45, 1, 1),
    }
    if side not in boxes:
        raise ValueError("Unknown orientation")
    a, b, c, d = boxes[side]
    return image.crop((int(a * w), int(b * h), int(c * w), int(d * h)))


def linear(spec, x):
    logits = np.asarray(spec["coef"]) @ x + np.asarray(spec["intercept"])
    return spec["classes"][int(logits.argmax())]


def merge_evidence(baseline, tractor, parts, candidates):
    disagreement = baseline["suggestion"] != tractor or baseline["suggestion"] != parts
    conflict = bool(
        candidates and (len(candidates) != 1 or baseline["suggestion"] not in candidates)
    )
    result = dict(baseline)
    if conflict or (baseline["status"] == "experimental_match" and disagreement):
        result.update(
            status="manual_review",
            model=None,
            family=None,
            reason="Crop/front-region disagreement or conflicting OCR evidence.",
        )
    return dict(result=result, branch_disagreement=disagreement, marking_conflict=conflict)
