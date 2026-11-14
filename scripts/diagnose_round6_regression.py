"""Factorial regression diagnosis on fixed features and the same protected control."""

import json
from pathlib import Path

import numpy as np

from scripts.benchmark_tractor_parts import fit, read_rows
from scripts.train_round5_recognition import evaluation_label, summarize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/continuation_round6"
SIX = {"mtz_80", "mtz_82", "mtz_820", "mtz_1221", "nh_t5000", "claas_axion_800"}


def features():
    full = {}
    for name in [
        "output/targeted_weather2/fused/features.npz",
        "output/continuation_round4/field_pilot/features.npz",
    ]:
        z = np.load(ROOT / name)
        full.update(dict(zip(z["keys"], z["values"], strict=True)))
    z = np.load(ROOT / "output/tractor_parts/features.npz")
    crops = {
        str(k).removesuffix("/tractor"): v
        for k, v in zip(z["keys"], z["values"], strict=True)
        if str(k).endswith("/tractor")
    }
    z = np.load(ROOT / "output/continuation_round5/features.npz")
    for k, v in zip(z["keys"], z["values"], strict=True):
        if str(k).startswith("full/"):
            full[str(k)[5:]] = v
        else:
            crops[str(k)[5:]] = v
    return full, crops


def main():
    oldrows, _, _ = read_rows()
    oldtrain = [r for r in oldrows if r["role"] == "train"]
    r4 = json.loads(
        (ROOT / "output/continuation_round4/field_pilot/dataset_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    r5 = json.loads(
        (ROOT / "output/continuation_round5/dataset_manifest.json").read_text(encoding="utf-8")
    )
    test = r5["test"]
    full, crops = features()
    joined = {k: np.concatenate((v, crops[k])) for k, v in full.items() if k in crops}
    variants = {
        "old_six_343_crop": (oldtrain, joined),
        "joint_369_crop": (r4["train"], joined),
        "joint_425_crop": (r5["train"], joined),
        "six_425_crop": (
            [dict(r, model=r["model"] if r["model"] in SIX else "other") for r in r5["train"]],
            joined,
        ),
        "joint_425_full": (r5["train"], full),
        "joint_425_duplicate_full": (
            r5["train"],
            {k: np.concatenate((v, v)) for k, v in full.items()},
        ),
    }
    result = {}
    for name, (train, vectors) in variants.items():
        m = fit(train, np.stack([vectors[r["key"]] for r in train]))
        p = m.predict(np.stack([vectors[r["key"]] for r in test]))
        records = [
            dict(
                key=r["key"],
                group=r["group"],
                cohort=r["cohort"],
                expected=evaluation_label(r["model"], m.classes_),
                prediction=str(pred),
            )
            for r, pred in zip(test, p, strict=True)
        ]
        result[name] = dict(
            summary=summarize(records),
            records=records,
            train=len(train),
            classes=m.classes_.tolist(),
        )
        print(
            name,
            {
                k: f"{v['correct']}/{v['total']}"
                for k, v in result[name]["summary"].items()
                if k != "by_class"
            },
            flush=True,
        )
    previous = json.loads(
        (ROOT / "output/continuation_round5/comparison.json").read_text(encoding="utf-8")
    )["runs"]["previous_six_full_crop"]
    assert [r["prediction"] for r in result["old_six_343_crop"]["records"]] == [
        r["prediction"] for r in previous
    ]
    (OUT / "regression_ablation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
