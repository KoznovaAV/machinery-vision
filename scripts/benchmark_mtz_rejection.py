"""Compare MTZ rejection after adding manually verified examples of model 80."""

from __future__ import annotations

import json

import torch

from scripts.benchmark_grouped_recognition import digest, embeddings, encoder, read_csv
from scripts.train_open_set_filter import ARTIFACT, ROOT, filter_scores, fit_filter

TRAIN_GROUPS = {"tractorbg-100353", "tractorbg-101761"}
VALIDATION_GROUPS = {"tractorbg-43150"}
POSITIVE_VALIDATION = {
    "tractorbg-110228",
    "tractorbg-149208",
    "tractorbg-108598",
    "tractorbg-63069",
}


def main():
    base = ARTIFACT.parent
    output = ROOT / "output/tractor_localization"
    old = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    weight = ROOT / "weights/dinov2-small.safetensors"
    if old["encoder_sha256"] != digest(weight):
        raise ValueError("Existing filter does not match the DINOv2 weights.")
    files = {}
    rows = []

    def add(r, directory, label, known, split):
        file = directory / r["file"]
        if r["id"] in files or digest(file) != r["sha256"]:
            raise ValueError(f"Duplicate id or changed file: {file}")
        files[r["id"]] = file
        rows.append(dict(id=r["id"], group=r["group"], label=label, known=known, split=split))

    expansion = ROOT / "output/mtz_expansion"
    for r in read_csv(expansion / "reviewed_manifest.csv"):
        split = {"fresh_train": "train", "fresh_test": "test", "stress_snow": "snow"}[r["subset"]]
        if r["group"] in POSITIVE_VALIDATION:
            split = "val"
        add(r, expansion, r["model"], True, split)
    for r in read_csv(base / "reviewed_candidates.csv"):
        if r["scope"] == "other_mtz":
            add(r, base, r["category"], False, r["split"])
    for r in read_csv(output / "reviewed_other_mtz.csv"):
        group = r["group"]
        split = (
            "train" if group in TRAIN_GROUPS else "val" if group in VALIDATION_GROUPS else "test"
        )
        add(r, output, "mtz_80", False, split)

    seen = {}
    for row in rows:
        if seen.setdefault(row["group"], row["split"]) != row["split"]:
            raise ValueError(f"Group crosses splits: {row['group']}")

    torch.set_num_threads(6)
    vectors = embeddings(encoder("dinov2", weight), files, pad=True)
    candidate = fit_filter(vectors, rows)
    test = [r for r in rows if r["split"] in {"test", "snow"}]
    results = {}
    for key, spec in (("existing", old["filters"]["mtz"]), ("candidate", candidate)):
        results[key] = []
        for row in test:
            scores = filter_scores(vectors[row["id"]], spec)
            results[key].append(
                {
                    **row,
                    "scores": scores,
                    "accept": {k: score >= spec["threshold"][k] for k, score in scores.items()},
                }
            )
    artifact = dict(
        specs=candidate,
        encoder_sha256=digest(weight),
        training_manifest_sha256=digest(output / "reviewed_other_mtz.csv"),
        partitions=rows,
        results=results,
    )
    (output / "mtz_rejection.json").write_text(
        json.dumps(artifact, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
