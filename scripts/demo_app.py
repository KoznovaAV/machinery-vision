#!/usr/bin/env python3
"""Gradio-демо Tractor Vision.

Переиспользует инференс-ядро ``src/api/main.py``: :func:`predict_image` и
приватную :func:`src.api.main._decide_state` с теми же порогами
``confidence_threshold``/``state_dirty_threshold`` из ``configs/config.yaml`` — логика
принятия решений о ``needs_review`` и ``dirty`` не дублируется.

Вкладка «Распознавание» — drag-drop фото и галерея примеров из
``demo_photos/``. Вкладка «VIN (демо-поток)» — поиск подмодели и года по VIN в
``data/vin_lookup.json``; VIN вне таблицы помечается как требующий оператора.

Запуск::

    python scripts/demo_app.py
    python scripts/demo_app.py --checkpoint weights/multi_task_best.ckpt
    python scripts/demo_app.py --share
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Позволяет запускать напрямую (`python scripts/demo_app.py`), а не только
# как модуль (`python -m scripts.demo_app`): без этого `src` не виден, т.к.
# sys.path[0] — это scripts/, а не корень проекта.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import gradio as gr  # noqa: E402
from PIL import Image  # noqa: E402

from src.api.main import CONFIDENCE_THRESHOLD, _decide_state  # noqa: E402
from src.config.classes import MODEL_CLASSES  # noqa: E402
from src.config.config_loader import load_config  # noqa: E402
from src.data.transforms import get_val_transforms  # noqa: E402
from src.models.loader import load_multi_task_model  # noqa: E402
from src.models.multi_task import MultiTaskTractorClassifier  # noqa: E402
from src.models.predict import predict_image  # noqa: E402
from src.models.registry import _checkpoint_sha, build_registry, get_model_meta  # noqa: E402

DEMO_PHOTOS_DIR: Path = Path("demo_photos")
VIN_LOOKUP_PATH: Path = Path("data/vin_lookup.json")
DEFAULT_REGISTRY_MODEL: str = "machine"


def _load_vin_lookup(path: Path) -> dict[str, dict[str, Any]]:
    """Загрузить демо-таблицу VIN -> подмодель/год.

    Args:
        path: Путь к JSON-файлу таблицы.

    Returns:
        Словарь ``VIN -> {family, submodel, year}``. Пустой словарь, если файл
        отсутствует.
    """
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_demo_model(
    checkpoint: Path | None, model_name: str
) -> tuple[MultiTaskTractorClassifier, str | None, str]:
    """Загрузить модель демо: чекпоинт-кандидат либо запись реестра.

    Args:
        checkpoint: Путь к чекпоинту-кандидату (флаг ``--checkpoint``) для
            репетиции без смены ``configs/config.yaml``; ``None`` — грузить из реестра.
        model_name: Имя записи реестра (``configs/config.yaml:models``), используется
            только если ``checkpoint`` не задан.

    Returns:
        Кортеж ``(модель в режиме eval, метка версии, sha чекпоинта)``.

    Raises:
        SystemExit: Если ``model_name`` отсутствует в реестре.
    """
    if checkpoint is not None:
        model = load_multi_task_model(checkpoint)
        return model, "candidate (--checkpoint)", _checkpoint_sha(checkpoint)

    registry = build_registry(load_config())
    if model_name not in registry:
        raise SystemExit(
            f"Неизвестная модель реестра: {model_name!r}. Доступны: {sorted(registry)}."
        )
    return get_model_meta(registry[model_name])


def build_interface(
    model: MultiTaskTractorClassifier,
    transform: Any,
    vin_table: dict[str, dict[str, Any]],
    model_version: str | None,
    checkpoint_sha: str,
) -> gr.Blocks:
    """Собрать Gradio-интерфейс из двух вкладок.

    Args:
        model: Загруженная multi-task модель в режиме eval.
        transform: Валидационная трансформация Albumentations.
        vin_table: Таблица VIN -> подмодель/год.
        model_version: Метка версии модели для шапки интерфейса.
        checkpoint_sha: Хеш чекпоинта для шапки интерфейса.

    Returns:
        Собранный, но ещё не запущенный ``gr.Blocks``.
    """

    def predict_fn(image: Image.Image | None) -> tuple[str, str, str, str, str]:
        """Прогнать фото через демо-модель и собрать поля интерфейса."""
        if image is None:
            return "", "", "", "", ""

        model_idx, confidence, state_idx, state_conf = predict_image(model, image, transform)
        state, state_confidence = _decide_state(state_idx, state_conf)
        family = MODEL_CLASSES[model_idx]

        banner = ""
        if confidence < CONFIDENCE_THRESHOLD:
            banner = (
                '<div style="background:#fff3cd;color:#664d03;padding:10px 14px;'
                'border-radius:6px;border:1px solid #ffe69c;">'
                f"⚠️ Низкая уверенность ({confidence:.0%}) — нужна проверка человеком."
                "</div>"
            )
        return family, f"{confidence:.1%}", state, f"{state_confidence:.1%}", banner

    def vin_lookup_fn(vin: str) -> str:
        """Найти подмодель и год по VIN в демо-таблице."""
        record = vin_table.get(vin.strip().upper())
        if record is None:
            return "VIN не найден, требуется оператор."
        return (
            f"Семейство: {record['family']}\n"
            f"Подмодель: {record['submodel']}\n"
            f"Год выпуска: {record['year']}"
        )

    with gr.Blocks(title="Tractor Vision — демо") as demo:
        gr.Markdown(
            f"# Tractor Vision — демо\n"
            f"Модель: `{model_version or 'н/д'}` · чекпоинт: `{checkpoint_sha}`"
        )

        with gr.Tab("Распознавание"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="pil", label="Фото трактора (drag-drop)")
                with gr.Column():
                    family_out = gr.Textbox(label="Семейство", interactive=False)
                    confidence_out = gr.Textbox(label="Confidence", interactive=False)
                    state_out = gr.Textbox(label="Состояние", interactive=False)
                    state_confidence_out = gr.Textbox(label="State confidence", interactive=False)
                    banner_out = gr.HTML()

            outputs = [family_out, confidence_out, state_out, state_confidence_out, banner_out]
            image_input.change(fn=predict_fn, inputs=image_input, outputs=outputs)

            if DEMO_PHOTOS_DIR.is_dir():
                examples = sorted(p for p in DEMO_PHOTOS_DIR.iterdir() if p.is_file())
                if examples:
                    gr.Examples(
                        examples=[[str(p)] for p in examples],
                        inputs=image_input,
                        outputs=outputs,
                        fn=predict_fn,
                        cache_examples=False,
                        run_on_click=True,
                        label="Примеры (клик = предсказание)",
                    )

        with gr.Tab("VIN (демо-поток)"):
            vin_input = gr.Textbox(label="VIN")
            vin_output = gr.Textbox(label="Результат", interactive=False, lines=3)
            vin_input.submit(fn=vin_lookup_fn, inputs=vin_input, outputs=vin_output)
            gr.Button("Найти", variant="primary").click(
                fn=vin_lookup_fn, inputs=vin_input, outputs=vin_output
            )

    return demo


def main(argv: list[str] | None = None) -> int:
    """Разобрать аргументы, загрузить модель и запустить Gradio-демо."""
    parser = argparse.ArgumentParser(
        description="Gradio-демо Tractor Vision: распознавание трактора + VIN-поиск."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Чекпоинт-кандидат для репетиции без смены configs/config.yaml.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_REGISTRY_MODEL,
        help=(
            "Имя записи реестра моделей (configs/config.yaml:models); "
            "игнорируется при --checkpoint."
        ),
    )
    parser.add_argument("--share", action="store_true", help="Создать публичную ссылку Gradio.")
    parser.add_argument("--port", type=int, default=7860, help="Порт локального сервера.")
    args = parser.parse_args(argv)

    config = load_config()
    model, model_version, checkpoint_sha = _load_demo_model(args.checkpoint, args.model)
    transform = get_val_transforms(config.image_size)
    vin_table = _load_vin_lookup(VIN_LOOKUP_PATH)

    print(f"Модель: {model_version or 'н/д'} · чекпоинт: {checkpoint_sha}")
    if not vin_table:
        print(
            f"Таблица VIN не найдена ({VIN_LOOKUP_PATH}) — вкладка VIN всегда вернёт «не найден»."
        )

    demo = build_interface(model, transform, vin_table, model_version, checkpoint_sha)
    demo.launch(server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    sys.exit(main())
