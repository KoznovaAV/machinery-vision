"""Compare the previously trained full-frame/crop head with marking-based ranking."""

import json

import numpy as np
import torch

from scripts.analyze_model_marking import revised_suggestion
from scripts.benchmark_supervised_parts import BASE, ROOT, fused
from scripts.benchmark_tractor_conditions import CLASSES

OUT = ROOT / "output/continuation_round4"


def predict(head, x):
    return head["classes"][
        int((np.asarray(head["coef"]) @ x + np.asarray(head["intercept"])).argmax())
    ]


def main():
    torch.set_num_threads(5)
    comparison = json.loads((OUT / "marking_comparison.json").read_text(encoding="utf-8"))
    old = json.loads(
        (ROOT / "output/targeted_weather2/fused/metrics.json").read_text(encoding="utf-8")
    )
    sources = {r["key"]: ROOT / r["path"] for r in old["partitions"]}
    prior = json.loads((ROOT / "reports/axle_marking_new_control.json").read_text(encoding="utf-8"))
    fresh = json.loads((OUT / "new_control.json").read_text(encoding="utf-8"))
    future = json.loads(
        (ROOT / "reports/future_field_model_registry.json").read_text(encoding="utf-8")
    )
    sources.update({r["key"]: ROOT / r["path"] for r in prior + fresh})
    sources.update(
        {"future/" + r["id"]: ROOT / r["path"] for r in future if r["split"] == "future_test"}
    )
    files = {}
    for r in comparison["predictions"]:
        if r["cohort"] == "legacy":
            continue
        name = r["key"].split("/")[-1]
        crop = (
            (ROOT / "output/axle_round2/new_views" / name / "tractor.jpg")
            if r["cohort"] == "prior_external"
            else OUT / "new_views" / name / "tractor.jpg"
        )
        files[r["key"] + "/full"] = sources[r["key"]]
        files[r["key"] + "/tractor"] = crop
    vectors = fused(files)
    fullcache = np.load(BASE / "features.npz")
    full = dict(zip(fullcache["keys"], fullcache["values"], strict=True))
    cropcache = np.load(ROOT / "output/tractor_parts/features.npz")
    machine = dict(zip(cropcache["keys"], cropcache["values"], strict=True))
    head = json.loads((ROOT / "output/axle_round2/comparison.json").read_text(encoding="utf-8"))[
        "heads"
    ]["full_tractor"]
    for r in comparison["predictions"]:
        a = full[r["key"]] if r["cohort"] == "legacy" else vectors[r["key"] + "/full"]
        b = (
            machine[r["key"] + "/tractor"]
            if r["cohort"] == "legacy"
            else vectors[r["key"] + "/tractor"]
        )
        label = predict(head, np.concatenate((a, b)))
        r["full_tractor_suggestion"] = label
        r["full_tractor_ranked"] = revised_suggestion(label, r["evidence"], CLASSES)
    summary = {}
    for cohort in sorted({r["cohort"] for r in comparison["predictions"]}):
        rr = [r for r in comparison["predictions"] if r["cohort"] == cohort]
        summary[cohort] = {}
        for subset in ["all", "supported", "outside"]:
            selected = [
                r
                for r in rr
                if subset == "all" or ((r["expected"] in CLASSES) == (subset == "supported"))
            ]
            summary[cohort][subset] = dict(
                total=len(selected),
                base=sum(r["baseline"]["suggestion"] == r["expected"] for r in selected),
                base_marking=sum(r["ranked_suggestion"] == r["expected"] for r in selected),
                full_tractor=sum(r["full_tractor_suggestion"] == r["expected"] for r in selected),
                full_tractor_marking=sum(
                    r["full_tractor_ranked"] == r["expected"] for r in selected
                ),
            )
    comparison["head_ablation"] = summary
    (OUT / "marking_comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
