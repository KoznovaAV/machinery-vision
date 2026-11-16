"""Project-relative defaults; callers may override caches through environment variables."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("MACHINERY_VISION_CACHE", ROOT / ".cache"))
for key, value in {
    "TORCH_HOME": ROOT / "weights/torch-cache",
    "HF_HOME": CACHE / "huggingface",
    "MPLCONFIGDIR": CACHE / "matplotlib",
    "YOLO_CONFIG_DIR": CACHE / "ultralytics",
}.items():
    os.environ.setdefault(key, str(value))
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
CONVNEXT_WEIGHTS = Path(os.environ["TORCH_HOME"]) / "hub/checkpoints/convnext_tiny-983f1562.pth"
