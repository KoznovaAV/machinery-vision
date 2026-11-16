"""Compare two-view model-marking hints against the frozen model without training on controls."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

from scripts.analyze_model_marking import inspect, revised_suggestion
from scripts.benchmark_grouped_recognition import digest
from scripts.benchmark_supervised_parts import fused, prepare_new
from scripts.benchmark_tractor_conditions import CLASSES
from scripts.benchmark_tractor_parts import BASE, ROOT, read_rows
from scripts.predict_tractor_conditions import decide

OUT = ROOT / "output/continuation_round4"


def main():
    torch.set_num_threads(5)
    import scripts.benchmark_supervised_parts as parts

    parts.OUT = OUT
    old, meta, head = read_rows()
    old = [dict(r, cohort="legacy") for r in old if r["role"] == "test"]
    prior = json.loads((ROOT / "reports/axle_marking_new_control.json").read_text(encoding="utf-8"))
    prior = [dict(r, cohort="prior_external") for r in prior]
    fresh = json.loads((OUT / "new_control.json").read_text(encoding="utf-8"))
    fresh = [dict(r, cohort="round4_external") for r in fresh]
    future = json.loads(
        (ROOT / "reports/future_field_model_registry.json").read_text(encoding="utf-8")
    )
    outside = [
        dict(
            key="future/" + r["id"],
            group=r["group"],
            path=r["path"],
            sha256=r["sha256"],
            model=r["model"],
            cohort="future_outside",
            role="test",
        )
        for r in future
        if r["split"] == "future_test"
    ]
    manifest_path = OUT / "evaluated_manifest.json"
    if manifest_path.exists():
        frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
        old = [r for r in frozen if r["cohort"] == "legacy"]
        prior = [r for r in frozen if r["cohort"] == "prior_external"]
        fresh = [r for r in frozen if r["cohort"] == "round4_external"]
        outside = [r for r in frozen if r["cohort"] == "future_outside"]
    else:
        manifest_path.write_text(
            json.dumps(old + prior + fresh + outside, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    for r in old + prior + fresh + outside:
        if digest(ROOT / r["path"]) != r["sha256"]:
            raise ValueError("Changed frozen control source")
    prepare_new(fresh + outside)
    cache = np.load(BASE / "features.npz")
    vectors = dict(zip(cache["keys"], cache["values"], strict=True))
    missing = prior + fresh + outside
    vectors.update(fused({r["key"]: ROOT / r["path"] for r in missing}))
    rows = old + prior + fresh + outside
    if len({r["sha256"] for r in rows}) != len(rows):
        raise ValueError("Repeated image in marking control")
    training = {r["group"] for r in head["training"]}
    if training & {r["group"] for r in rows}:
        raise ValueError("Control group in classifier training")
    engine = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1)
    directory = OUT / "marking"
    directory.mkdir(exist_ok=True)
    predictions = []
    for i, r in enumerate(rows):
        if r["cohort"] == "legacy":
            path = ROOT / "output/tractor_parts/views" / r["sha256"] / "tractor.jpg"
        elif r["cohort"] == "prior_external":
            path = ROOT / "output/axle_round2/new_views" / r["key"].split("/")[-1] / "tractor.jpg"
        else:
            path = OUT / "new_views" / r["key"].split("/")[-1] / "tractor.jpg"
        target = directory / (r["sha256"] + ".json")
        if target.exists():
            evidence = json.loads(target.read_text(encoding="utf-8"))
        else:
            with Image.open(path) as im:
                evidence = inspect(im, engine)
            target.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        baseline = decide(vectors[r["key"]], head)
        expected = r["model"] if r["model"] in CLASSES else "other"
        ranked = revised_suggestion(baseline["suggestion"], evidence, CLASSES)
        accepted = baseline["status"] == "experimental_match"
        new_accepted = accepted and not (
            evidence["status"] == "consistent_marking" and ranked != baseline["suggestion"]
        )
        predictions.append(
            dict(
                key=r["key"],
                group=r["group"],
                cohort=r["cohort"],
                expected=expected,
                original_label=r["model"],
                baseline=baseline,
                ranked_suggestion=ranked,
                evidence=evidence,
                baseline_exact=accepted,
                marking_guard_exact=new_accepted,
            )
        )
        print("marking", i + 1, "/", len(rows), r["key"], evidence["status"], flush=True)
    summary = {}
    for cohort in sorted({r["cohort"] for r in predictions}):
        rr = [r for r in predictions if r["cohort"] == cohort]
        summary[cohort] = dict(
            total=len(rr),
            baseline_correct=sum(r["baseline"]["suggestion"] == r["expected"] for r in rr),
            ranked_correct=sum(r["ranked_suggestion"] == r["expected"] for r in rr),
            consistent=sum(r["evidence"]["status"] == "consistent_marking" for r in rr),
            baseline_exact_correct=sum(
                r["baseline_exact"] and r["baseline"]["suggestion"] == r["expected"] for r in rr
            ),
            baseline_exact_wrong=sum(
                r["baseline_exact"] and r["baseline"]["suggestion"] != r["expected"] for r in rr
            ),
            guard_exact_correct=sum(
                r["marking_guard_exact"] and r["baseline"]["suggestion"] == r["expected"]
                for r in rr
            ),
            guard_exact_wrong=sum(
                r["marking_guard_exact"] and r["baseline"]["suggestion"] != r["expected"]
                for r in rr
            ),
        )
    result = dict(
        summary=summary,
        predictions=predictions,
        head_sha256=digest(BASE / "expanded.json"),
        script_sha256=digest(Path(__file__)),
        marking_script_sha256=digest(ROOT / "scripts/analyze_model_marking.py"),
        policy="Hints only; no exact MTZ upgrades. Two views share image pixels.",
    )
    (OUT / "marking_comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
