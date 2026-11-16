"""Train a small parts localizer on manually entered source-group-safe boxes."""

import hashlib
import json
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/axle_round2"


def validate_annotations(rows):
    selected = [r for r in rows if r["role"] == "train"]
    if len({r["key"] for r in selected}) != len(selected):
        raise ValueError("Repeated training key")
    train_groups = {r["group"] for r in selected if r["index"] not in {1, 14, 24}}
    val_groups = {r["group"] for r in selected if r["index"] in {1, 14, 24}}
    if train_groups & val_groups:
        raise ValueError("Detector train/validation group leakage")
    for row in selected:
        for box in row["boxes"]:
            x1, y1, x2, y2 = box["xyxy"]
            if not 0 <= x1 < x2 <= 1 or not 0 <= y1 < y2 <= 1:
                raise ValueError("Invalid normalized annotation box")
    return selected


def main():
    from ultralytics import YOLO

    rows = json.loads((ROOT / "reports/axle_marking_annotations.json").read_text(encoding="utf-8"))
    rows = validate_annotations(rows)
    labels = {"front_axle": 0, "model_marking": 1}
    for r in rows:
        if r["role"] != "train":
            continue
        split = "val" if r["index"] in {1, 14, 24} else "train"
        name = "old-" + str(r["index"])
        imdir = OUT / "yolo/images" / split
        lbdir = OUT / "yolo/labels" / split
        imdir.mkdir(parents=True, exist_ok=True)
        lbdir.mkdir(parents=True, exist_ok=True)
        if hashlib.sha256((ROOT / r["path"]).read_bytes()).hexdigest() != r["sha256"]:
            raise ValueError("Changed annotated image")
        shutil.copyfile(ROOT / r["path"], imdir / (name + ".jpg"))
        lines = []
        for b in r["boxes"]:
            x1, y1, x2, y2 = b["xyxy"]
            if not 0 <= x1 < x2 <= 1 or not 0 <= y1 < y2 <= 1:
                raise ValueError("Invalid manual box")
            lines.append(f'{labels[b["label"]]} {(x1+x2)/2} {(y1+y2)/2} {x2-x1} {y2-y1}')
        (lbdir / (name + ".txt")).write_text("\n".join(lines), encoding="utf-8")
    spec = dict(
        path=(OUT / "yolo").as_posix(),
        train="images/train",
        val="images/val",
        names={v: k for k, v in labels.items()},
    )
    (OUT / "yolo/data.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    model = YOLO(ROOT / "weights/yolo11n.pt")
    model.train(
        data=str(OUT / "yolo/data.yaml"),
        epochs=40,
        imgsz=416,
        batch=4,
        device="cpu",
        workers=0,
        project=str(OUT),
        name="detector_v2",
        exist_ok=True,
        seed=83,
        deterministic=True,
        optimizer="AdamW",
        lr0=0.001,
        freeze=10,
        patience=100,
        plots=False,
        mosaic=0.5,
        mixup=0,
        fliplr=0.5,
        amp=False,
        save=True,
        verbose=False,
    )
    print("Fixed last.pt checkpoint; no control images used for checkpoint selection.")


if __name__ == "__main__":
    main()
