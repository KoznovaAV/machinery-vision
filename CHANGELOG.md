# Changelog

Восстановлен по истории исходного репозитория (47 коммитов, 2026-07-06 …
2026-09-14); коммиты, не относящиеся к коду, данным и технической документации,
не включены.

## 2026-10-20 — подготовка репозитория

- `config.yaml` перенесён в `configs/config.yaml`; путь по умолчанию в
  `src/config/config_loader.py`, `Dockerfile`, `docker-compose.yml` и документах обновлён.
- Model card перенесена в `reports/`; добавлены `reports/weights_inventory.md`,
  отчёты разбора ошибок состояния, логи экспериментов в `docs/experiments/`.
- Добавлены `docs/PROJECT_CONTEXT.md`, `docs/labeling_rules.md`,
  `docs/known_issues.md`, `docs/decisions.md`.
- Убраны исторические формулировки из docstring и комментариев.

## 2026-09-14

- docs: `RETRAIN.md` синхронизирован с состоянием после v3.

## 2026-09-11

- chore: промоут v3 в реестр моделей (clean false-dirty 0.128 → 0.098).
- feat: Gradio-демо с потоком поиска по VIN (`scripts/demo_app.py`).

## 2026-09-02

- feat: нейтральные аугментации снега, «мыла» камеры и бликов в
  `generate_dirty_dataset` (применяются к `clean` и `dirty`); усилено смещение
  `mud_crust` в нижнюю зону кадра; порог видимости синтетической грязи
  `MIN_DIRT_INTENSITY=0.18`; расширены диапазон lighting-деградации и train-jitter.
- feat: `scripts/check_images.py` — проверка целостности JPG/PNG.

## 2026-09-01

- feat: `model_version` и `checkpoint_sha` в ответах API.
- feat: эндпоинт `/predict_batch` с пофайловой обработкой ошибок.
- feat: аутентификация по `X-API-Key` и пер-ключевой rate limit.
- ci: GitHub Actions (pre-commit + pytest).

## 2026-08-31

- feat: реестр моделей, управляемый `config.models`.
- feat: `/feedback` и флаг `needs_review` по `confidence_threshold`;
  `scripts/ingest_feedback.py` с дедупликацией и режимом `--dry-run`.
- feat: разбор ошибок головы состояния (`scripts/error_analysis_state.py`),
  `state_confidence` в API, свип порога `--sweep`.
- feat: `--init-checkpoint` для дообучения с тёплого старта; чтение манифестов с BOM.
- chore: промоут fine-tune v2, `state_dirty_threshold = 0.60`.
- fix: каталог feedback вынесен в docker volume.
- refactor: удалён single-task конвейер; общие загрузчик чекпоинта и инференс
  одного изображения; `eval_visualize` слит в `multi_task_evaluate`; Docker CMD по
  умолчанию — API; канонические имена классов в источниках данных.
- docs: README переписан под 4 класса, создан `docs/`; метрики после чистки
  данных; расширены model card и описание классов.

## 2026-08-30

- feat: дообучение на реальной грязи — dirty recall 0.375 → 0.906;
  исправлен выбор чекпоинта в evaluate; добавлен `scripts/collect_dirty_tractors.py`.

## 2026-08-29

- feat: конфиг классов, скрипты сбора датасета, 4-классовый конвейер, загрузчик
  конфига, partial fine-tuning, инструменты пересборки датасета.
- refactor: `mtz_82` → `mtz_belarus` (слитое семейство 82 / 1221).
- feat: расширены поисковые запросы коллектора.
- test: тесты датасета и модели приведены к 4-классовому контракту.

## 2026-07-06 … 2026-07-10

- Начальная структура проекта и первая версия ML-системы.
