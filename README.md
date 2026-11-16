# Machinery Vision

<p align="center">
  <img alt="Python 3.11" src="https://img.shields.io/badge/PYTHON-3.11-00ADD8?style=for-the-badge&amp;logo=python&amp;logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PYTORCH-CPU-61DAFB?style=for-the-badge&amp;logo=pytorch&amp;logoColor=white">
  <img alt="Grounding DINO" src="https://img.shields.io/badge/GROUNDING%20DINO-1D9BF0?style=for-the-badge">
  <img alt="FastAPI" src="https://img.shields.io/badge/FASTAPI-FF6B00?style=for-the-badge&amp;logo=fastapi&amp;logoColor=white">
</p>

MVP распознавания сельхозтехники: **тип → семейство → кандидаты модели**. Выделяет машины, проверяет прицепное оборудование, читает маркировку и оценивает видимые признаки грязи/снега. При недостатке данных отказывается от ответа. Точные модификации автоматически не подтверждаются.

## Запуск

Python 3.11, Git. Папка проекта может находиться на любом диске; пути и кэши по умолчанию вычисляются относительно неё.

```bash
git clone https://github.com/KoznovaAV/machinery-vision.git
cd machinery-vision
python -m venv .venv
```

Активируйте окружение: Windows PowerShell — `.\.venv\Scripts\Activate.ps1`, Linux/macOS — `source .venv/bin/activate`. Далее одинаковые команды:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m scripts.setup_weights
python -m scripts.check_release
python -m scripts.web
```

Открыть **http://127.0.0.1:7864/**. JPEG/PNG/WebP до 8 МБ, минимум 64×64. Другой порт: `python -m scripts.web --port 7865`. Остановка — `Ctrl+C`. Первая обработка загружает модели и занимает больше времени.

`setup_weights` скачивает около 0,9 ГБ базовых весов и проверяет SHA-256. Головы, пороги и эталоны уже включены в Git. Интернет для распознавания после подготовки не нужен. Полностью автономный Windows-комплект с Python, пакетами и датасетом передаётся отдельно; команды — в [инструкции передачи](docs/HANDOVER.md).

По умолчанию веса — `weights/`, кэши — `.cache/`, результаты — `runs/`. При необходимости задайте `TORCH_HOME` или `MACHINERY_VISION_CACHE`. Специальный путь к пользовательскому профилю или определённый диск не требуется.

## Возможности и ограничения

В справочнике восемь семейств. Ответ о семействе разрешён для МТЗ-80/82/820, МТЗ-1221, New Holland T5000, CLAAS AXION 800 и ДТ-75. МТЗ-1523, Т-150К и ACROS пока выдаются кандидатами. T5000/AXION 800 — серии. Обычный результат: «семейство известно; точная модель неясна».

Рабочие веса при организации проекта не переобучались и не заменялись кандидатами. На сохранённом расчёте голов с OCR: **103 верных и 2 ошибочных принятых ответа на 158 фото**, ещё 53 отказа. Это не общая точность приложения. Зимний срез хуже: 4 верных и 3 ошибочных ответа на 12 фото. Подробности по каждому классу и состояниям — в [карточке модели](docs/MODEL_CARD.md).

Текущая оценка состояния подтверждает признаки **грязи** или **снега в кадре**. «Чистый», дождь и мокрый корпус сейчас не подтверждаются. Снег на фоне не доказывает снег на машине. Ночь, перекрытие и спорные области могут приводить к отказу. Прицепы ещё могут путаться с машинами; 20 проверенных сцен не доказывают отсутствие ошибок на новых фото.

## Проверки и обучение

```bash
python -m scripts.check_release
python -m pytest tests -q
```

Данные из отдельного комплекта положите в `datasets/`. После этого доступны:

```bash
python -m scripts.audit_data
python -m scripts.evaluate
python -m scripts.train --task conditions
python -m scripts.train --task families
python -m scripts.train --task regions
```

Обучение создаёт отдельного кандидата в `runs/training/`. Рабочий комплект `artifacts/mvp/` не заменяется. Добавление проверенных фото и признаков описано в [правилах данных](docs/DATA.md) и [инструкции обучения](docs/TRAINING.md).

## Состав проекта

| Папка | Назначение |
|---|---|
| `src/models/` | Только текущие детектор, энкодеры, иерархия, детали, маркировка и состояние |
| `src/training/` | Групповое обучение и сравнение кандидатов |
| `scripts/` | Запуск, загрузка весов, аудит, признаки, обучение, проверка |
| `configs/` | Единый справочник типов/семейств/моделей |
| `artifacts/mvp/` | Один актуальный комплект, включая область переднего моста |
| `evaluation/` | Численные результаты, без дневников экспериментов |
| `datasets/` | Фото, разметка, контроль и кэши признаков; отдельно от Git |
| `weights/` | Большие базовые веса; отдельно от Git |
| `runs/` | Новые результаты и кандидаты; отдельно от Git |

Документация: [модель](docs/MODEL_CARD.md), [данные](docs/DATA.md), [обучение](docs/TRAINING.md), [передача](docs/HANDOVER.md). Старые API, Gradio, Docker-конфигурация другого конвейера и дневники раундов убраны из текущего дерева. История Git сохранена.
