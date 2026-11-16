# Контекст API v3

Актуальная иерархическая модель и её датасеты описаны в [карте проекта](CURRENT.md).

Распознавание семьи трактора по фото и его состояния (`clean` / `dirty`).
FastAPI + PyTorch Lightning + ConvNeXt-Tiny, одна сеть с двумя головами
(4 семьи + состояние).

## Окружение

- ОС: Windows 11, PowerShell.
- Python 3.11. Подготовка окружения описана в [README](../README.md).
- GPU: NVIDIA RTX 4050 Laptop (CUDA) — для обучения; инференс и CI работают на CPU.
- Все команды выполняются из корня репозитория.

## Правила разработки

1. Классы и размеры — только из `src/config/classes.py` и `configs/config.yaml`.
   Хардкод запрещён.
2. Новые скрипты: `argparse`, type hints, docstring (Google-style, на русском).
3. Перед коммитом: `pytest tests/ -q` зелёный, pre-commit хуки проходят.
4. Не удалять и не переименовывать `data/*` и `weights/*` без явного решения.
5. Крупные правки — в отдельной ветке, не в `main`.
6. Коммиты атомарные: `feat:`, `fix:`, `refactor:`, `docs:`, `chore:`.
7. Стиль: black / isort / flake8 по `setup.cfg`. Комментарии — только «почему».
   Редактируешь файл — обновляй его docstring: только текущее поведение.

## Карта кода

| Путь | Назначение |
|------|------------|
| `src/config/` | источник правды: `classes.py`, `config_loader.py` |
| `src/data/` | датасет, трансформы, деградации (`dataset.py`, `transforms.py`, `generate_dirty_dataset.py`) |
| `src/data/utils.py` | общие утилиты конвейеров данных (`compute_content_hash`) |
| `src/models/` | архитектура (`multi_task.py`), загрузка чекпоинта (`loader.py`), инференс одного изображения (`predict.py`) |
| `src/models/registry.py` | реестр моделей инференса из `config.models` |
| `src/training/` | обучение и оценка (`multi_task_train.py`, `multi_task_evaluate.py`) |
| `src/api/` | FastAPI (`main.py`, `schemas.py`) |
| `scripts/` | конвейеры: сбор, подготовка, генерация, цикл real-dirty, демо |
| `scripts/error_analysis_state.py` | разбор промахов головы состояния; `--sweep` калибрует `state_dirty_threshold` по dirty recall на `data/real_dirty_val` |
| `scripts/ingest_feedback.py` | вливание пользовательского фидбэка в train |
| `tests/` | тесты API, модели и конвейера данных, включая проверки с локальными данными и весами |
| `configs/config.yaml` | единый конфиг |
| `.github/workflows/ci.yml` | GitHub Actions: pre-commit + pytest на CPU-torch |

## Текущее состояние

Снимок на 2026-09-01 (последнее обновление проектных заметок), с поправкой
на промоут v3 от 2026-09-11 — подробнее в [known_issues.md](known_issues.md).

- 4 класса: `chtz`, `johndeere`, `kirovets`, `mtz_belarus` (без подмоделей).
- Метрики модели v3 и состав оценочных выборок — [в model card](../reports/MODEL_CARD.md).
- `state_dirty_threshold`: 0.60 (калибровка `error_analysis_state.py --sweep`).
- API: `/health`, `/models` (реестр из `config.models`), `/predict`
  (`request_id`, `needs_review`, `model_version` / `checkpoint_sha`),
  `/predict_batch` (пофайловая обработка ошибок), `/feedback`.
- Аутентификация: заголовок `X-API-Key`, ключи в переменной окружения
  `TRACTOR_VISION_API_KEYS` (через запятую); при пустой переменной
  аутентификация отключена.
- Rate limit: пер-ключ, in-memory окно (настройки в `configs/config.yaml`).
- Обучение: `--init-checkpoint` (дообучение с готовых весов),
  `--early-stopping-patience` (ранняя остановка по val).
- Docker: CPU-образ собирался, но устарел — нужна пересборка.

Реестр весов и их роли — [reports/weights_inventory.md](../reports/weights_inventory.md).
