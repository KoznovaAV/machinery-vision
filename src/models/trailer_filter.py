"""Conservative semantic veto for attached trailers among multiple machine proposals."""

from src.models.machinery_regions import iou

PROMPT = "tractor. combine harvester. bulldozer. trailer. agricultural implement."


def powered_label(label):
    return any(word in label.lower() for word in ("tractor", "harvester", "bulldozer"))


def split_trailers(
    regions, semantic, *, minimum=0.35, overlap=0.70, margin=0.10, require_other_powered=True
):
    """Veto an aligned trailer only if another powered machine has positive support.

    Broad boxes containing a tractor and trailer never match a trailer by containment
    alone. Ambiguous labels and truncated tractors remain available to the verifier.
    Scores are detector evidence, not calibrated probabilities.
    """
    if not regions or (len(regions) < 2 and require_other_powered):
        return regions, []
    evidence = []
    for region in regions:
        aligned = [r for r in semantic if iou(region["box"], r["box"]) >= overlap]
        positive = max((r["score"] for r in aligned if powered_label(r["label"])), default=0.0)
        negative = max(
            (
                r["score"]
                for r in aligned
                if "trailer" in r["label"].lower() and not powered_label(r["label"])
            ),
            default=0.0,
        )
        evidence.append((positive, negative))
    kept = []
    ignored = []
    for index, (region, (positive, negative)) in enumerate(zip(regions, evidence, strict=True)):
        other_powered = any(j != index and p >= minimum for j, (p, n) in enumerate(evidence))
        if (
            (other_powered or not require_other_powered)
            and negative >= minimum
            and negative - positive >= margin
        ):
            ignored.append(
                dict(
                    bbox=region["box"],
                    reason="semantic_trailer",
                    verification=dict(
                        policy="aligned_trailer_v1",
                        positive_score=positive,
                        trailer_score=negative,
                        minimum=minimum,
                        overlap=overlap,
                        margin=margin,
                    ),
                )
            )
        else:
            kept.append(region)
    return kept, ignored


def _area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _intersection(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def remove_composite_views(kept, confirmed_attachments):
    """Remove a weaker union box around a stronger machine and confirmed trailer.

    Requires nearly complete union coverage and spatially distinct child regions;
    ordinary overlapping machines and ambiguous attachment proposals are preserved.
    """
    removed = []
    drop = set()
    for parent_index, parent in kept:
        p = parent["box"]
        ap = _area(p)
        if not ap:
            continue
        for child_index, child in kept:
            if child_index == parent_index or child["score"] < parent["score"] + 0.03:
                continue
            m = child["box"]
            am = _area(m)
            if not 0.35 <= am / ap <= 0.80 or _intersection(p, m) / max(am, 1) < 0.95:
                continue
            for attachment in confirmed_attachments:
                if attachment.get("reason") != "semantic_trailer":
                    continue
                a = attachment["bbox"]
                aa = _area(a)
                if (
                    not 0.20 <= aa / ap <= 0.75
                    or _intersection(p, a) / max(aa, 1) < 0.95
                    or iou(m, a) > 0.15
                ):
                    continue
                union = (_intersection(p, m) + _intersection(p, a) - _intersection(m, a)) / ap
                if union >= 0.90:
                    drop.add(parent_index)
                    removed.append(
                        dict(
                            bbox=p,
                            reason="composite_duplicate",
                            machine_region_id=child_index,
                            attachment_bbox=a,
                            union_coverage=union,
                        )
                    )
                    break
            if parent_index in drop:
                break
    return [(i, r) for i, r in kept if i not in drop], removed
