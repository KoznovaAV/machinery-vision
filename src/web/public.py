"""Bounded public uploads, image-only artifacts and expiring request data."""

from __future__ import annotations

import html
import asyncio
import logging
import re
import shutil
import threading
import time
from pathlib import Path

from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

MAX_BODY_BYTES = 8_100_000
RETENTION_SECONDS = 20 * 60


def error_page(message: str, status_code: int):
    return HTMLResponse(
        '<!doctype html><html lang="ru"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Проверка фотографии</title><body><main>"
        f'<p>{html.escape(message)}</p><a href="/">Загрузить фотографию</a>'
        "</main></body></html>",
        status_code=status_code,
    )


class PublicRequestLimits:
    """Accept one bounded prediction request per server process."""

    def __init__(self, app, max_body_bytes=MAX_BODY_BYTES):
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.slot = threading.BoundedSemaphore(1)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def response_send(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.extend(
                    [
                        (b"cache-control", b"no-store"),
                        (b"x-content-type-options", b"nosniff"),
                        (b"x-robots-tag", b"noindex, nofollow"),
                        (b"referrer-policy", b"no-referrer"),
                        (
                            b"content-security-policy",
                            b"default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'",
                        ),
                    ]
                )
                message = dict(message, headers=headers)
            await send(message)

        if scope["method"] != "POST" or scope["path"] != "/predict":
            return await self.app(scope, receive, response_send)
        if not self.slot.acquire(blocking=False):
            return await error_page(
                "Сейчас обрабатывается другая фотография. Повторите запрос через несколько секунд.",
                429,
            )(scope, receive, response_send)
        try:
            headers = dict(scope.get("headers", []))
            try:
                declared = int(headers.get(b"content-length", b"0"))
            except ValueError:
                return await error_page("Некорректный размер запроса.", 400)(
                    scope, receive, response_send
                )
            if declared > self.max_body_bytes:
                return await error_page("Фото должно быть меньше 8 МБ.", 413)(
                    scope, receive, response_send
                )
            body = bytearray()
            started = time.monotonic()
            while True:
                try:
                    message = await asyncio.wait_for(
                        receive(), timeout=max(0.001, 60 - (time.monotonic() - started))
                    )
                except TimeoutError:
                    return await error_page(
                        "Загрузка заняла слишком много времени. Повторите запрос.", 408
                    )(scope, receive, response_send)
                if message["type"] == "http.disconnect":
                    return
                body.extend(message.get("body", b""))
                if len(body) > self.max_body_bytes:
                    return await error_page("Фото должно быть меньше 8 МБ.", 413)(
                        scope, receive, response_send
                    )
                if not message.get("more_body", False):
                    break
            sent = False

            async def bounded_receive():
                nonlocal sent
                if not sent:
                    sent = True
                    return {"type": "http.request", "body": bytes(body), "more_body": False}
                return await receive()

            await self.app(scope, bounded_receive, response_send)
        finally:
            self.slot.release()


class PublicImages(StaticFiles):
    async def get_response(self, path, scope):
        parts = Path(path).parts
        if not parts or not re.fullmatch(r"[0-9a-f]{32}", parts[0]):
            return error_page("Файл недоступен.", 404)
        folder = Path(self.directory) / parts[0]
        if not folder.is_dir() or time.time() - folder.stat().st_mtime > RETENTION_SECONDS:
            return error_page("Срок хранения результата истёк.", 404)
        if Path(path).suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            return error_page("Файл недоступен.", 404)
        return await super().get_response(path, scope)


def purge_expired(directory, active=(), *, now=None):
    """Remove expired UUID request folders within the designated upload directory."""
    root = Path(directory).resolve()
    now = time.time() if now is None else now
    for child in root.iterdir():
        if not re.fullmatch(r"[0-9a-f]{32}", child.name) or child in active:
            continue
        if child.resolve().parent != root or not child.is_dir() or child.is_symlink():
            continue
        if now - child.stat().st_mtime > RETENTION_SECONDS:
            try:
                shutil.rmtree(child)
            except OSError:
                logging.getLogger(__name__).warning(
                    "Could not remove expired request %s", child.name
                )
