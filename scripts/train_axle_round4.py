"""Train the expanded axle-only localizer; keep the previous control splits fixed."""

import hashlib
import json
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO

from scripts.train_tractor_parts_detector import validate_annotations

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/continuation_round4"


def main():
    rows = json.loads((ROOT / "reports/round4_axle_annotations.json").read_text(encoding="utf-8"))
    training = validate_annotations(rows)
    controls = {r["group"] for r in rows if r["role"] == "test"}
    if {r["group"] for r in training} & controls:
        raise ValueError("Control group in detector training")
    for r in training:
        source = ROOT / r["path"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != r["sha256"]:
            raise ValueError("Changed annotation source")
        split = "val" if r["index"] in {1, 14, 24} else "train"
        name = "frame-" + str(r["index"])
        img = OUT / "yolo_axle/images" / split
        lab = OUT / "yolo_axle/labels" / split
        img.mkdir(parents=True, exist_ok=True)
        lab.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, img / (name + ".jpg"))
        lines = []
        for b in r["boxes"]:
            if b["label"] != "front_axle":
                continue
            a, c, d, e = b["xyxy"]
            lines.append(f"0 {(a+d)/2} {(c+e)/2} {d-a} {e-c}")
        (lab / (name + ".txt")).write_text("\n".join(lines), encoding="utf-8")
    spec = dict(
        path=(OUT / "yolo_axle").as_posix(),
        train="images/train",
        val="images/val",
        names={0: "front_axle"},
    )
    file = OUT / "yolo_axle/data.yaml"
    file.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    model = YOLO(ROOT / "weights/yolo11n.pt")
    model.train(
        data=str(file),
        epochs=40,
        imgsz=640,
        batch=4,
        device="cpu",
        workers=0,
        project=str(OUT),
        name="axle_detector",
        exist_ok=True,
        seed=83,
        deterministic=True,
        optimizer="AdamW",
        lr0=0.001,
        freeze=None,
        patience=100,
        plots=False,
        mosaic=0.2,
        mixup=0,
        fliplr=0.5,
        amp=False,
        save=True,
        verbose=False,
    )
    print("Evaluation uses last.pt; no test-dependent checkpoint selection.")


if __name__ == "__main__":
    main()
