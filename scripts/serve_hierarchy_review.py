"""Local upload interface for the experimental hierarchy; production API is separate."""

import argparse
import logging
import threading
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from scripts.render_hierarchy_review import render
from src.config.hierarchy_artifacts import hierarchy_bundle_path

ROOT = Path(__file__).resolve().parents[1]


def create_app(directory=None, bundle=None, predictor=None):
    directory = Path(directory or ROOT / "output/hierarchy_review").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    bundle = Path(bundle) if bundle is not None else hierarchy_bundle_path()
    lock = threading.Lock()
    app = FastAPI(title="Machinery hierarchy review")
    app.mount("/artifacts", StaticFiles(directory=directory), name="artifacts")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return render(dict(objects=[], initial=True))

    @app.post("/predict", response_class=HTMLResponse)
    async def predict(file: UploadFile = File(...)):
        payload = await file.read(8_000_001)
        if len(payload) > 8_000_000:
            raise HTTPException(413, "Фото должно быть меньше 8 МБ")
        try:
            with Image.open(BytesIO(payload)) as im:
                if im.format not in {"JPEG", "PNG", "WEBP"}:
                    raise ValueError("Unsupported image format")
                if min(im.size) < 64:
                    raise ValueError("Изображение слишком маленькое")
                im.verify()
        except (OSError, ValueError, UnidentifiedImageError):
            raise HTTPException(400, "Нужно корректное фото JPEG, PNG или WebP") from None
        folder = directory / uuid4().hex
        folder.mkdir()
        source = folder / "input.jpg"
        source.write_bytes(payload)
        output = folder / "prediction.json"

        def work():
            function = predictor
            if function is None:
                from scripts.predict_machinery_hierarchy import run

                function = run
            with lock:
                return function(source, bundle, output)

        try:
            result = await run_in_threadpool(work)
        except Exception:
            logging.getLogger(__name__).exception("Hierarchy inference failed")
            raise HTTPException(
                500,
                "Не удалось обработать фото. Проверьте наличие обученных весов и журнал сервера.",
            ) from None
        page = render(result, directory)
        (folder / "review.html").write_text(page, encoding="utf-8")
        return page

    return app


def main():
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7862)
    args = parser.parse_args()
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
