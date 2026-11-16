"""Photo upload interface for machinery recognition."""

from src.config.runtime import ROOT

import argparse
import asyncio
from contextlib import asynccontextmanager, suppress
import logging
import threading
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from scripts.render_review import render
from src.config.hierarchy_artifacts import hierarchy_bundle_path
from src.web.public import PublicImages, PublicRequestLimits, purge_expired, error_page

ROOT = Path(__file__).resolve().parents[1]


def create_app(directory=None, bundle=None, predictor=None, *, public_demo=False):
    directory = Path(
        directory or ROOT / ("runs/public-demo" if public_demo else "runs/review")
    ).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    bundle = Path(bundle) if bundle is not None else hierarchy_bundle_path()
    lock = threading.Lock()
    active = set()

    @asynccontextmanager
    async def lifespan(app):
        task = None
        if public_demo:

            async def cleanup():
                while True:
                    purge_expired(directory, active)
                    await asyncio.sleep(60)

            task = asyncio.create_task(cleanup())
        try:
            yield
        finally:
            if task:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    app = FastAPI(
        title="Machinery Vision",
        lifespan=lifespan,
        docs_url=None if public_demo else "/docs",
        redoc_url=None if public_demo else "/redoc",
        openapi_url=None if public_demo else "/openapi.json",
    )
    files = PublicImages if public_demo else StaticFiles
    app.mount("/artifacts", files(directory=directory), name="artifacts")
    if public_demo:
        app.add_middleware(PublicRequestLimits)

        @app.exception_handler(HTTPException)
        async def public_error(request, exc):
            return error_page(str(exc.detail), exc.status_code)

    @app.get("/health")
    def health():
        return {"status": "ok", "public_demo": public_demo}

    @app.get("/", response_class=HTMLResponse)
    def index():
        return render(dict(objects=[], initial=True), public_demo=public_demo)

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
                if public_demo and (im.width * im.height > 16_000_000 or max(im.size) > 8192):
                    raise HTTPException(
                        413,
                        "Для демо уменьшите фото до 16 мегапикселей и 8192 пикселей по стороне.",
                    )
                im.verify()
        except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError):
            raise HTTPException(400, "Нужно корректное фото JPEG, PNG или WebP") from None
        if public_demo:
            folders = list(directory.iterdir())
            stored = sum(
                p.stat().st_size for d in folders if d.is_dir() for p in d.rglob("*") if p.is_file()
            )
            if len(folders) >= 40 or stored >= 256 * 1024 * 1024:
                raise HTTPException(429, "Достигнут временный лимит демо. Повторите запрос позже.")
        folder = directory / uuid4().hex
        folder.mkdir()
        source = folder / ("input.png" if public_demo else "input.jpg")
        if not public_demo:
            source.write_bytes(payload)
        output = folder / "prediction.json"

        def work():
            if public_demo:
                with Image.open(BytesIO(payload)) as original:
                    clean = ImageOps.exif_transpose(original).convert("RGB")
                    clean.info.clear()
                    clean.save(source, format="PNG")
            function = predictor
            if function is None:
                from scripts.predict import run

                function = run
            with lock:
                return function(source, bundle, output)

        active.add(folder)
        try:
            result = await run_in_threadpool(work)
        except Exception:
            logging.getLogger(__name__).exception("Hierarchy inference failed")
            raise HTTPException(
                500,
                "Не удалось обработать фото. Проверьте наличие обученных весов и журнал сервера.",
            ) from None
        finally:
            active.discard(folder)
        page = render(result, directory, public_demo=public_demo)
        (folder / "review.html").write_text(page, encoding="utf-8")
        return page

    return app


def main():
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7864)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--public-demo", action="store_true")
    args = parser.parse_args()
    uvicorn.run(create_app(public_demo=args.public_demo), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
