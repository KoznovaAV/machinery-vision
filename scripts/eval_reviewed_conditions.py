"""Evaluate frozen local classifiers against a reviewed manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scripts.benchmark_grouped_recognition import digest, embeddings, encoder, read_csv
from scripts.compare_synthetic_dirt import reference_scores


def candidate_scores(head_file, weight_file, paths, name):
    head = json.loads(head_file.read_text(encoding="utf-8"))
    if name == "dinov2" and digest(weight_file) != head["encoder_sha256"]:
        raise ValueError("DINOv2 weights do not match the saved head.")
    vectors = embeddings(encoder(name, weight_file), paths, pad=name == "dinov2")
    scores = {
        key: float(vector @ head["coef"]) + head["intercept"] for key, vector in vectors.items()
    }
    probs = {key: float(0.5 * (1 + np.tanh(score / 2))) for key, score in scores.items()}
    return probs, head.get("threshold", 0.5), head["classes"]


def summary(rows, threshold):
    slices = defaultdict(list)
    for row in rows:
        tag = "snow" if row["weather"].startswith("snow_") else row["weather"]
        slices[tag].append(row)
        if row["surface"] == "immersed":
            slices["water_contact"].append(row)
        slices[row["scope"]].append(row)
    result = {}
    for name, data in slices.items():
        result[name] = {"images": len(data), "groups": len({item["group"] for item in data})}
        for name_model, cutoff in (("candidate", threshold), ("v3", 0.60)):
            pred = [item for item in data if item[f"{name_model}_p_dirty"] >= cutoff]
            result[name][name_model] = {
                "marked_dirty": len(pred),
                "dirty_detected": sum(item["mud"] == "dirty" for item in pred),
                "false_dirty_visible_clean": sum(item["mud"] == "clean" for item in pred),
                "dirty_labeled": sum(item["mud"] == "dirty" for item in data),
                "clean_labeled": sum(item["mud"] == "clean" for item in data),
            }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(5)
    torch.set_num_interop_threads(1)
    manifest = read_csv(args.manifest)
    roots = {"data": args.project, "photos": args.manifest.parent}
    paths = {}
    for row in manifest:
        if row["status"] not in {"included", "pending_overlap"}:
            continue
        key = row["id"]
        path = roots[row["path"].split("/")[0]] / row["path"]
        if digest(path) != row["sha256"]:
            raise ValueError(f"Unexpected SHA-256 for {key}")
        paths[key] = path
    selected = [row for row in manifest if row["status"] == "included"]
    heads = args.project / "output/dataset_audit_next"
    weights = args.project / "weights"
    mtz_items = [row for row in selected if row["model"]]
    scores, cutoff, classes = candidate_scores(
        heads / "mtz_candidate.json",
        weights / "dinov2-small.safetensors",
        {row["id"]: paths[row["id"]] for row in mtz_items},
        "dinov2",
    )
    groups = defaultdict(list)
    for row in mtz_items:
        score = scores[row["id"]]
        row["mtz_pred"] = classes[int(score >= cutoff)]
        row["mtz_score_82"] = score
        groups[row["group"]].append(row["mtz_pred"] == row["model"])
    state_inputs = {row["id"]: paths[row["id"]] for row in selected}
    scores, threshold, _ = candidate_scores(
        heads / "state_candidate.json", None, state_inputs, "convnext"
    )
    reference, _ = reference_scores(
        weights / "multi_task_best_v3.ckpt", list(state_inputs.values()), "cpu"
    )
    for (key, score_v3), row in zip(
        zip(state_inputs, reference, strict=True), selected, strict=True
    ):
        row["candidate_p_dirty"] = scores[key]
        row["v3_p_dirty"] = float(score_v3)
    result = {
        "manifest_rows": len(manifest),
        "status": dict(Counter(r["status"] for r in manifest)),
        "mtz": {
            "images": len(mtz_items),
            "images_correct": sum(sum(values) for values in groups.values()),
            "groups": len(groups),
            "groups_all_views_correct": sum(all(values) for values in groups.values()),
            "counts": dict(Counter(r["model"] for r in mtz_items)),
        },
        "state_candidate_threshold": threshold,
        "conditions": summary(selected, threshold),
        "per_image": selected,
    }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "per_image"}, indent=2))


if __name__ == "__main__":
    main()
