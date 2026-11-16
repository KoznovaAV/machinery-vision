"""Summarize the fixed tractor-parts comparison without threshold tuning."""

import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/tractor_parts"


def main():
    d = json.loads((OUT / "comparison.json").read_text(encoding="utf-8"))
    rows = d["predictions"]
    previous = json.loads(
        (ROOT / "output/targeted_weather2/fused/metrics.json").read_text(encoding="utf-8")
    )
    details = {r["key"]: r for r in previous["runs"]["expanded"]}
    for r in rows:
        for field in ("condition", "view", "source"):
            r[field] = details[r["key"]].get(field, "unknown")
    known = [r for r in rows if r["expected"] in d["summary"]["baseline"]["by_class"]]
    summary = d["summary"]
    target = {"mtz_80", "mtz_820", "nh_t5000"}
    for name, s in summary.items():
        s["focus_correct"] = sum(s["by_class"][x]["correct"] for x in target)
        s["focus_total"] = sum(s["by_class"][x]["total"] for x in target)

        def pred(r):
            v = r[name]
            return v["prediction"] if isinstance(v, dict) else v

        s["cross_focus_errors"] = sum(
            pred(r) in target and pred(r) != r["expected"] for r in known if r["expected"] in target
        )
        s["focus_other_errors"] = sum(
            pred(r) not in target for r in known if r["expected"] in target
        )
        s["fixed"] = sum(pred(r) == r["expected"] and r["baseline"] != r["expected"] for r in known)
        s["broken"] = sum(
            pred(r) != r["expected"] and r["baseline"] == r["expected"] for r in known
        )
        # Paired bootstrap by independent source group; diagnostic, not new validation.
        groups = sorted({r["group"] for r in known})
        rng = np.random.default_rng(83)
        changes = {
            g: [
                int(pred(r) == r["expected"]) - int(r["baseline"] == r["expected"])
                for r in known
                if r["group"] == g
            ]
            for g in groups
        }
        samples = []
        for _ in range(2000):
            picked = rng.choice(groups, len(groups), replace=True)
            vals = [v for g in picked for v in changes[g]]
            samples.append(sum(vals) / len(vals))
        s["paired_group_bootstrap_delta_95"] = np.quantile(samples, [0.025, 0.975]).tolist()
    d["operational"] = {}
    for name, field in [
        ("baseline", "baseline_exact"),
        ("ocr_veto", "baseline_with_ocr_veto_exact"),
        ("parts_agreement", "agreement_exact"),
    ]:
        selected = [r for r in rows if r.get(field)]
        d["operational"][name] = dict(
            correct=sum(r["baseline"] == r["expected"] for r in selected),
            wrong=sum(r["baseline"] != r["expected"] for r in selected),
            accepted=len(selected),
            review=len(rows) - len(selected),
        )
    for name, use_crop, use_front in [("crop_ocr", True, False), ("front_ocr", False, True)]:
        selected = [
            r
            for r in rows
            if r["baseline_with_ocr_veto_exact"]
            and (not use_crop or r["tractor"]["prediction"] == r["baseline"])
            and (not use_front or r["full_tractor_front"] == r["baseline"])
        ]
        d["operational"][name] = dict(
            correct=sum(r["baseline"] == r["expected"] for r in selected),
            wrong=sum(r["baseline"] != r["expected"] for r in selected),
            accepted=len(selected),
            review=len(rows) - len(selected),
        )
    d["ocr_summary"] = dict(
        with_text=sum(bool(r["ocr"]["lines"]) for r in rows),
        with_candidate=sum(bool(r["ocr"]["candidates"]) for r in rows),
        candidates=[
            dict(key=r["key"], expected=r["expected"], **r["ocr"])
            for r in rows
            if r["ocr"]["candidates"]
        ],
    )
    annotations = json.loads(
        (ROOT / "reports/tractor_parts_manual_review.json").read_text(encoding="utf-8")
    )
    d["manual_review"] = dict(
        total=len(annotations), visibility=dict(Counter(r["axle_visibility"] for r in annotations))
    )
    (OUT / "comparison.json").write_text(
        json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (ROOT / "reports/tractor_parts_metrics.json").write_text(
        json.dumps(
            {k: v for k, v in d.items() if k != "predictions"}, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            dict(summary=summary, operational=d["operational"], ocr=d["ocr_summary"]),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
