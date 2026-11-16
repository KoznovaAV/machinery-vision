"""Run the local recognition and operator review interface."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel, Field, ValidationError
from starlette.concurrency import run_in_threadpool

from scripts.benchmark_grouped_recognition import digest, embeddings, encoder
from scripts.train_open_set_filter import ARTIFACT, filter_scores
from src.config.classes import MODEL_CLASSES
from src.data.transforms import get_val_transforms
from src.models.loader import load_multi_task_model

ROOT = Path(__file__).resolve().parent.parent
PILOT = ROOT / "output/dataset_audit_next"
WEIGHTS = ROOT / "weights"
REVIEWS_DB = ROOT / "output/operator_review/reviews.sqlite3"
MAX_BYTES = 12 * 1024 * 1024
MAX_PIXELS = 30_000_000
app = FastAPI(title="Machinery Vision · локальная проверка")
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])


class Review(BaseModel):
    model: str = Field(pattern=r"^(mtz_82|mtz_1221|other|unknown)$")
    mud: str = Field(pattern=r"^(clean|dirty|unknown)$")
    condition: str = Field(pattern=r"^(dry|wet|snow|unknown)$")
    note: str = Field(default="", max_length=1000)


async def read_upload(file: UploadFile) -> tuple[bytes, Image.Image]:
    content = await file.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES:
        raise HTTPException(413, "Размер фото превышает 12 МБ.")
    try:
        with Image.open(io.BytesIO(content)) as original:
            if original.format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError("unsupported format")
            if original.width * original.height > MAX_PIXELS:
                raise ValueError("too many pixels")
            return content, ImageOps.exif_transpose(original).convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(422, "Нужен читаемый JPEG, PNG или WebP до 30 Мп.") from exc


@lru_cache(maxsize=1)
def load_models():
    torch.set_num_threads(4)
    dino = WEIGHTS / "dinov2-small.safetensors"
    prior = json.loads((PILOT / "mtz_candidate.json").read_text(encoding="utf-8"))
    state = json.loads((PILOT / "state_candidate.json").read_text(encoding="utf-8"))
    next_head = ROOT / "output/photo_pilot/mtz_experimental.json"
    weighted_head = ROOT / "output/photo_pilot/mtz_experimental_weighted.json"
    heads = {"mtz": prior, "state": state}
    for key in ("expanded", "expanded_group_balanced"):
        path = ROOT / "output/mtz_expansion" / (key + ".json")
        if path.is_file():
            heads["mtz_" + key] = json.loads(path.read_text(encoding="utf-8"))
    filter_spec = json.loads(ARTIFACT.read_text(encoding="utf-8")) if ARTIFACT.is_file() else None
    if filter_spec and filter_spec["encoder_sha256"] != digest(dino):
        raise ValueError("DINOv2 weights do not match the experimental filter.")
    if next_head.is_file():
        heads["mtz_new"] = json.loads(next_head.read_text(encoding="utf-8"))
    if weighted_head.is_file():
        heads["mtz_weighted"] = json.loads(weighted_head.read_text(encoding="utf-8"))
    if any(
        head["encoder_sha256"] != digest(dino)
        for name, head in heads.items()
        if name.startswith("mtz")
    ):
        raise ValueError("DINOv2 weights do not match the saved heads.")
    return (
        load_multi_task_model(WEIGHTS / "multi_task_best_v3.ckpt"),
        encoder("dinov2", dino),
        encoder("convnext", None),
        heads,
        filter_spec,
    )


def score_head(features, head):
    logit = float(features @ head["coef"]) + head["intercept"]
    return 0.5 * (1.0 + math.tanh(logit / 2.0))


def predict(image: Image.Image) -> dict:
    reference, dino, convnext, heads, filter_spec = load_models()
    tensor = get_val_transforms()(image=np.asarray(image))["image"].unsqueeze(0)
    with torch.inference_mode():
        family, state = reference(tensor)
    model_family = MODEL_CLASSES[int(family.argmax(1).item())]
    v3_score = float(state.softmax(1)[0, 1])
    candidate_score = score_head(
        embeddings(convnext, {"upload": image}, pad=False)["upload"], heads["state"]
    )
    models = {}
    checks = {}
    if filter_spec or model_family == "mtz_belarus":
        vector = embeddings(dino, {"upload": image}, pad=True)["upload"]
    if filter_spec:
        for name, spec in filter_spec["filters"].items():
            scores = filter_scores(vector, spec)
            checks[name] = {
                method: {"score": score, "accepted": score >= spec["threshold"][method]}
                for method, score in scores.items()
            }
    if model_family == "mtz_belarus":
        for key, head in heads.items():
            if key.startswith("mtz"):
                p82 = score_head(vector, head)
                models[key] = {"model": head["classes"][int(p82 >= 0.5)], "p82": p82}
    return {
        "family": model_family,
        "family_score": float(family.softmax(1).max()),
        "filter": checks,
        "mtz": models,
        "v3": {"p_dirty": v3_score, "threshold": 0.6},
        "candidate": {"p_dirty": candidate_score, "threshold": heads["state"]["threshold"]},
    }


def review_decision(first: dict, second: dict | None = None) -> dict:
    views = (first, second) if second else (first,)
    if any(not view["filter"] for view in views):
        return {"status": "unavailable"}

    def passes(view, level):
        return view["filter"][level]["linear"]["accepted"]

    supported = [v for v in views if passes(v, "family")]
    if not supported:
        return {"status": "outside_classes"}
    if len(supported) != len(views) or len({v["family"] for v in views}) > 1:
        return {"status": "manual_review"}

    family = first["family"]
    if any(v["family_score"] < 0.60 for v in views):
        return {"status": "manual_review"}
    if family != "mtz_belarus":
        return {"status": "family_only", "family": family}

    if any(not passes(v, "mtz") for v in views):
        return {"status": "manual_review", "family": family}
    scores = [v["mtz"].get("mtz_expanded", v["mtz"]["mtz"])["p82"] for v in views]
    mean = sum(scores) / len(scores)
    if len({score >= 0.5 for score in scores}) != 1 or max(mean, 1 - mean) < 0.65:
        return {"status": "manual_review", "family": family}
    return {"status": "model", "family": family, "model": "mtz_82" if mean >= 0.5 else "mtz_1221"}


def predict_uploads(image: Image.Image, second: Image.Image | None) -> dict:
    first_prediction = predict(image)
    second_prediction = predict(second) if second is not None else None
    return {
        **first_prediction,
        "second": second_prediction,
        "decision": review_decision(first_prediction, second_prediction),
    }


@app.post("/api/predict")
async def recognition(file: UploadFile = File(...), second: UploadFile | None = File(None)):
    content, image = await read_upload(file)
    second_image = None
    if second is not None:
        second_content, second_image = await read_upload(second)
        if second_content == content:
            raise HTTPException(422, "Для сравнения нужен другой ракурс той же машины.")
    return {
        "sha256": hashlib.sha256(content).hexdigest(),
        **await run_in_threadpool(predict_uploads, image, second_image),
    }


def store_review(content: bytes, review: Review) -> str:
    digest_key = hashlib.sha256(content).hexdigest()
    REVIEWS_DB.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(REVIEWS_DB)) as db, db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS reviews "
            "(sha256 TEXT PRIMARY KEY, photo BLOB NOT NULL, model TEXT NOT NULL, "
            "mud TEXT NOT NULL, condition TEXT NOT NULL, "
            "note TEXT NOT NULL, reviewed_at TEXT NOT NULL)"
        )
        db.execute(
            "INSERT INTO reviews VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(sha256) DO UPDATE SET model=excluded.model, mud=excluded.mud, "
            "condition=excluded.condition, note=excluded.note, reviewed_at=excluded.reviewed_at",
            (
                digest_key,
                content,
                review.model,
                review.mud,
                review.condition,
                review.note,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
    return digest_key


@app.post("/api/review")
async def review_photo(request: Request, file: UploadFile = File(...), labels: str = Form(...)):
    origin = request.headers.get("origin")
    if origin and origin != f"{request.url.scheme}://{request.url.netloc}":
        raise HTTPException(403, "Сохранение доступно из локального интерфейса.")
    try:
        review = Review.model_validate_json(labels)
    except ValidationError as exc:
        raise HTTPException(422, "Проверьте значения меток и длину комментария.") from exc
    content, _ = await read_upload(file)
    return {"sha256": await run_in_threadpool(store_review, content, review)}


@app.get("/api/reviews.csv")
def export_reviews() -> Response:
    """Download saved operator labels as a spreadsheet-safe CSV."""
    buffer = io.StringIO(newline="")
    csv_file = csv.writer(buffer)
    csv_file.writerow(("sha256", "model", "mud", "condition", "note", "reviewed_at"))
    if REVIEWS_DB.is_file():
        with closing(sqlite3.connect(REVIEWS_DB)) as db:
            columns = "sha256, model, mud, condition, note, reviewed_at"
            for sha256, model, mud, condition, note, reviewed_at in db.execute(
                f"SELECT {columns} FROM reviews ORDER BY reviewed_at DESC"
            ):
                if note and note.lstrip().startswith(("=", "+", "-", "@")):
                    note = "'" + note
                csv_file.writerow((sha256, model, mud, condition, note, reviewed_at))
    return Response(
        buffer.getvalue().encode("utf-8-sig"),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="machinery-vision-reviews.csv"'},
    )


app.mount("/", StaticFiles(directory=ROOT / "scripts/review_web", html=True), name="review_web")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
