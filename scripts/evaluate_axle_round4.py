"""Evaluate the expanded axle detector on frozen manually annotated controls."""

import json
from pathlib import Path

from PIL import Image, ImageDraw
from ultralytics import YOLO

from scripts.benchmark_grouped_recognition import digest
from scripts.benchmark_tractor_localization import intersection_over_union

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/continuation_round4"


def main():
    rows = json.loads((ROOT / "reports/round4_axle_annotations.json").read_text(encoding="utf-8"))
    controls = [r for r in rows if r["role"] == "test"]
    model = YOLO(OUT / "axle_detector/weights/last.pt")
    results = []
    for r in controls:
        path = ROOT / r["path"]
        if digest(path) != r["sha256"]:
            raise ValueError("Changed annotated control")
        pred = model.predict(str(path), imgsz=640, conf=0.15, device="cpu", verbose=False)[0]
        boxes = pred.boxes.xyxy.tolist()
        scores = pred.boxes.conf.tolist()
        with Image.open(path) as im:
            w, h = im.size
        gold = [
            [v * (w if i % 2 == 0 else h) for i, v in enumerate(b["xyxy"])]
            for b in r["boxes"]
            if b["label"] == "front_axle"
        ]
        best = max((intersection_over_union(a, b) for a in gold for b in boxes), default=0.0)
        results.append(
            dict(
                key=r["key"],
                cohort=r.get("cohort", "legacy"),
                annotated=len(gold),
                predicted=len(boxes),
                best_iou=best,
                hit=bool(gold and best >= 0.5),
                boxes=boxes,
                scores=scores,
            )
        )
    summary = {}
    for co in sorted({r["cohort"] for r in results}):
        rr = [r for r in results if r["cohort"] == co]
        summary[co] = dict(
            annotated=sum(r["annotated"] > 0 for r in rr),
            hit=sum(r["hit"] for r in rr),
            unboxed_with_detection=sum(not r["annotated"] and bool(r["predicted"]) for r in rr),
        )
    out = dict(
        summary=summary,
        records=results,
        weights_sha256=digest(OUT / "axle_detector/weights/last.pt"),
        annotations_sha256=digest(ROOT / "reports/round4_axle_annotations.json"),
        protocol="Fixed last epoch, confidence .15, IoU .5; same manual controls as round2.",
    )
    (OUT / "axle_localization.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    for start in range(0, len(controls), 9):
        sheet = Image.new("RGB", (1350, 1050), "white")
        d = ImageDraw.Draw(sheet)
        for j, (r, pred) in enumerate(
            zip(controls[start : start + 9], results[start : start + 9], strict=True)
        ):
            im = Image.open(ROOT / r["path"]).convert("RGB")
            draw = ImageDraw.Draw(im)
            for b in pred["boxes"]:
                draw.rectangle(b, outline="red", width=max(2, im.width // 200))
            for b in r["boxes"]:
                if b["label"] == "front_axle":
                    draw.rectangle(
                        [
                            v * (im.width if i % 2 == 0 else im.height)
                            for i, v in enumerate(b["xyxy"])
                        ],
                        outline="lime",
                        width=max(2, im.width // 220),
                    )
            im.thumbnail((445, 310))
            x = j % 3 * 450
            y = j // 3 * 350
            sheet.paste(im, (x, y + 28))
            d.text(
                (x + 2, y + 2),
                r["key"].split("/")[-1] + f" IoU={pred['best_iou']:.2f}",
                fill="black",
            )
        sheet.save(OUT / f"axle-control-{start//9}.jpg")


if __name__ == "__main__":
    main()
