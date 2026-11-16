"""Evaluate marking hints on a frozen expanded control without training on its pixels."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps
from rapidocr_onnxruntime import RapidOCR

from scripts.analyze_model_marking import PROTOCOL, inspect, revised_suggestion
from scripts.train_round5_recognition import summarize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/continuation_round5"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    data = json.loads((OUT / "dataset_manifest.json").read_text(encoding="utf-8"))
    test = data["test"]
    prior = json.loads(
        (ROOT / "output/continuation_round4/marking_comparison.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (ROOT / "output/continuation_round4/evaluated_manifest.json").read_text(encoding="utf-8")
    )
    sha = {r["key"]: r["sha256"] for r in manifest}
    evidence = {sha[r["key"]]: r["evidence"] for r in prior["predictions"] if r["key"] in sha}
    oldsha = {
        r["key"]: r["sha256"]
        for r in json.loads(
            (ROOT / "output/continuation_round4/field_pilot/dataset_manifest.json").read_text(
                encoding="utf-8"
            )
        )["test"]
    }
    for r in json.loads(
        (ROOT / "output/continuation_round4/field_pilot/new_models_marking.json").read_text(
            encoding="utf-8"
        )
    ):
        evidence[oldsha[r["key"]]] = r["marking"]
    folder = OUT / "marking"
    folder.mkdir(exist_ok=True)
    engine = None
    for i, r in enumerate(test):
        file = folder / (r["sha256"] + ".json")
        if file.exists():
            evidence[r["sha256"]] = json.loads(file.read_text(encoding="utf-8"))
        if r["sha256"] not in evidence:
            if engine is None:
                engine = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1)
            with Image.open(ROOT / r["path"]) as im:
                evidence[r["sha256"]] = inspect(im, engine)
            print("marking", i + 1, "/", len(test), r["key"], flush=True)
        if evidence[r["sha256"]].get("protocol") != PROTOCOL:
            raise ValueError("Changed OCR protocol")
        file.write_text(
            json.dumps(evidence[r["sha256"]], ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if not args.compare:
        return
    comparison = json.loads((OUT / "comparison.json").read_text(encoding="utf-8"))
    for name, records in list(comparison["runs"].items()):
        if name == "previous_six_full_crop":
            supported = set(
                json.loads(
                    (ROOT / "output/axle_round2/comparison.json").read_text(encoding="utf-8")
                )["heads"]["full_tractor"]["classes"]
            )
        else:
            headfile = (
                (ROOT / "output/continuation_round4/field_pilot/expanded_head.json")
                if name == "previous_full"
                else OUT / (name + "_head.json")
            )
            supported = set(json.loads(headfile.read_text(encoding="utf-8"))["classes"])
        revised = []
        for r in records:
            reading = evidence[r["sha256"]]
            label = revised_suggestion(r["prediction"], reading, supported)
            revised.append(
                dict(
                    r,
                    prediction=label,
                    visual_prediction=r["prediction"],
                    marking=reading["suggestion"],
                    marking_status=reading["status"],
                )
            )
        comparison["runs"][name + "_marking"] = revised
        comparison["summary"][name + "_marking"] = summarize(revised)
    (OUT / "marking_comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    new = [
        r
        for r in test
        if r["key"].startswith("round5/")
        and evidence[r["sha256"]]["status"] == "consistent_marking"
    ]
    for start in range(0, len(new), 6):
        sheet = Image.new("RGB", (1500, 1050), "white")
        d = ImageDraw.Draw(sheet)
        for j, r in enumerate(new[start : start + 6]):
            im = ImageOps.exif_transpose(Image.open(ROOT / r["path"])).convert("RGB")
            im.thumbnail((495, 475))
            x = j % 3 * 500
            y = j // 3 * 525
            sheet.paste(im, (x, y + 35))
            reading = evidence[r["sha256"]]
            d.text((x, y), r["key"] + " " + str(reading["suggestion"]), fill="black")
        sheet.save(OUT / f"marking-review-{start//6}.jpg")
    print(json.dumps(comparison["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
