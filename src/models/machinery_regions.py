"""Keep distinct machinery instances while removing duplicate detector boxes."""

from __future__ import annotations


def iou(a, b):
    x = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    y = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = x * y

    def area(v):
        return max(0, v[2] - v[0]) * max(0, v[3] - v[1])

    union = area(a) + area(b) - inter
    return inter / union if union else 0.0


def centered_duplicate(a, b):
    """Recognize nested views of one object with almost identical box centres."""
    widths = [a[2] - a[0], b[2] - b[0]]
    heights = [a[3] - a[1], b[3] - b[1]]
    if min(widths + heights) <= 0:
        return False
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1]))
    containment = intersection / min(widths[0] * heights[0], widths[1] * heights[1])
    dx = abs((a[0] + a[2] - b[0] - b[2]) / 2) / min(widths)
    dy = abs((a[1] + a[3] - b[1] - b[3]) / 2) / min(heights)
    return containment >= 0.95 and dx <= 0.10 and dy <= 0.10


def select_regions(raw, width, height, max_objects=5):
    result = []
    for record in sorted(raw, key=lambda r: r["score"], reverse=True):
        a, b, c, d = record["box"]
        box = [max(0, int(a)), max(0, int(b)), min(width, int(c + 1)), min(height, int(d + 1))]
        if (
            box[2] - box[0] < 24
            or box[3] - box[1] < 24
            or (box[2] - box[0]) * (box[3] - box[1]) < 0.025 * width * height
        ):
            continue
        if any(iou(box, r["box"]) > 0.6 or centered_duplicate(box, r["box"])
               for r in result):
            continue
        label = record["label"].lower()
        kinds = []
        if "combine" in label or "harvester" in label:
            kinds.append("combine_harvester")
        if "bulldozer" in label:
            kinds.append("bulldozer")
        if "tractor" in label:
            kinds.append("tractor")
        if not kinds:
            continue
        kind = kinds[0] if len(kinds) == 1 else None
        result.append(dict(box=box, score=record["score"], type_candidate=kind))
        if len(result) >= max_objects:
            break
    return result
