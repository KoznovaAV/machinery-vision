# Переобучение API v3

Актуальная иерархическая модель и её датасеты описаны в [карте проекта](CURRENT.md).

Команды для Windows + PowerShell и Python 3.11 выполняются из корня проекта. GPU-обучение предполагает доступную CUDA.

Значения классов и размеров нигде не хардкодятся — берутся из
`src/config/classes.py` и `configs/config.yaml`.

## Фаза 0. Окружение

```powershell
# Активируйте окружение Python 3.11 по инструкции в README.
pip install -r requirements.txt

# классы согласованы с конфигом
python -c "from src.config.classes import MODEL_CLASSES; from src.config.config_loader import load_config; assert load_config().num_model_classes == len(MODEL_CLASSES) == 4; print('OK', MODEL_CLASSES)"
```

## Фаза 1. Данные: collect → prepare → generate

Строгий порядок, каждый шаг — вход следующего (подробности — `docs/DATA.md`).

```powershell
# 1a. сбор по классам (сначала малый объём для проверки выдачи)
python -m scripts.collect_tractors --output-dir data/collected --split train --per-class 20 --clip-threshold 0.7
# затем догнать до цели
python -m scripts.collect_tractors --output-dir data/collected --split train --per-class 600 --clip-threshold 0.7
```

Вручную разобрать `data/collected/**/to_review/` — валидные фото в классы,
мусор удалить.

```powershell
# 1b. объединение источников + стратифицированная пересборка 70/15/15
python -m scripts.prepare_dataset --sources data/processed data/collected --output-dir data/processed_rebuilt --ratios 0.7 0.15 0.15 --seed 42

# после проверки распределения в выводе — заменить рабочее дерево
Remove-Item -Recurse -Force data/processed
Rename-Item data/processed_rebuilt data/processed
```

```powershell
# 1c. синтетическая грязь поверх финального processed
python -m src.data.generate_dirty_dataset --clean-dir data/processed --output-dir data/dirty_clean --dirty-per-clean 2 --seed 42
```

Результат: `data/dirty_clean/{train,val,test}/<class>/{clean,dirty}/`. Просмотреть
5–10 грязных сэмплов глазами (грязь на технике, небо чистое).

## Фаза 2. Обучение multi-task

Начинать с **`--num-unfrozen-stages 1`**: на датасете умеренного размера
разморозка одной последней стадии backbone даёт меньший риск переобучения, чем
двух. Балансировка — `uncertainty` (стабильнее gradnorm).

```powershell
python -m src.training.multi_task_train `
    --data-dir data/dirty_clean `
    --num-unfrozen-stages 1 `
    --loss-balancing uncertainty `
    --image-size 384 `
    --max-epochs 100 `
    --accelerator gpu
```

В логах проверить строку `Разморожены стадии backbone (features): [...]` и что
LR backbone в 10 раз меньше LR голов (`LearningRateMonitor`).

Эскалация, только если `val_model_acc` или состояние явно проседают:

- `--num-unfrozen-stages 2` (больше ёмкости, выше риск переобучения);
- сравнить `--loss-balancing gradnorm` по `val_model_acc`.

Чекпоинты: `weights/multi-task-best-epoch=NN-val_model_acc=X.XXX.ckpt`
(по метрике) и `weights/multi_task_final.ckpt`. Рабочим файлом сделать лучший:

```powershell
$checkpoint = Read-Host "Путь к выбранному чекпоинту текущего прогона"
Copy-Item -LiteralPath $checkpoint -Destination weights/multi_task_best.ckpt
```

### Дообучение с тёплого старта

Для инкрементального дообучения поверх рабочего чекпоинта (например, после
вливания фидбэка) — вместо обучения с ImageNet-инициализации:

- `--init-checkpoint <path>` — загрузить веса модели из готового `.ckpt` перед
  обучением. Используется та же логика, что в
  `src/models/loader.py:load_multi_task_model` (снятие префикса `model.`,
  отбрасывание параметров балансировки `log_var_*`/`task_weights`); параметры
  балансировки текущего запуска стартуют с нуля.
- `--early-stopping-patience <N>` — терпение `EarlyStopping` по `val_model_acc`
  (по умолчанию `15`). Для короткого дообучения (`--max-epochs 6..8`) ставить
  `3..4`.
- `--lr` понизить относительно базового `5e-4` (ориентир `1e-4`; backbone
  остаётся в 10 раз медленнее).

```powershell
python -m src.training.multi_task_train `
    --data-dir data/dirty_clean `
    --init-checkpoint weights/multi_task_best.ckpt `
    --num-unfrozen-stages 1 `
    --lr 1e-4 `
    --max-epochs 8 `
    --early-stopping-patience 3 `
    --weights-dir weights/ft_v2 `
    --accelerator gpu
```

`--weights-dir` с отдельной папкой держит новые чекпоинты вне `weights/`, пока
кандидат не подтверждён.

> **ВАЖНО.** При дообучении ради состояния (v4 «грязь на ходовой» и т.п.):
> - early stopping и выбор best-чекпоинта сейчас идут по `val_model_acc`
>   (захардкожено в `multi_task_train.py`);
> - поэтому выбор эпохи по состоянию подтверждай вручную по `val_state_acc`
>   в логах по эпохам: лучший `val_model_acc` и лучший `val_state_acc` могут
>   приходиться на разные эпохи;
> - `--weights-dir weights/ft_state_vN` — чтобы не затереть best по семье;
> - финальная приёмка — только через `scripts/error_analysis_state.py` +
>   `--sweep`.
>
> Для состояния доступен флаг `--monitor val_state_acc`. По умолчанию используется
> `val_model_acc`. Выбранная метрика управляет сохранением лучшей эпохи, ранней
> остановкой и планировщиком LR. Для проверки реальной грязи используйте отдельные
> выборки и явный путь к чекпоинту.

## Фаза 3. Оценка

```powershell
python -m src.training.multi_task_evaluate --data-dir data/dirty_clean --image-size 384 --out-dir output
```

Оценка идёт по сплиту `val`. Сплит `test` в `data/dirty_clean` держится
нетронутым как финальный held-out — прогонять по нему один раз перед выкладкой,
не используя для подбора гиперпараметров.

Артефакты: `output/confusion_matrix.png` (4×4), `output/misclassified_examples.png`.

Целевые метрики (актуальные после v3):
- model acc (val) ≥ 0.995
- dirty recall (real_dirty_val) ≥ 0.90
- false-dirty (real_clean_probe) ≤ 0.15
- false-dirty (synthetic clean, порог 0.60) ≤ 0.10

## Фаза 4. Цикл дообучения на реальной грязи

Отдельный цикл — когда классы уже распознаются хорошо, но состояние проседает на
реальной экстремальной грязи. Цель: **dirty recall ≥ 0.90** на
`data/real_dirty_val`.

```powershell
# 4a. сбор реальных грязных фото
python -m scripts.collect_dirty_tractors --output-dir data/real_dirty_raw --limit 300 --clip-threshold 0.7

# 4b. псевдоразметка по классам (+ ручная проверка to_review/)
python -m scripts.pseudo_label_dirty --input-dir data/real_dirty_raw/unsorted --output-dir data/real_dirty_labeled --threshold 0.8

# 4c. разбиение 80/20: train-добавка + held-out val
python -m scripts.split_real_dirty --sources data/collected_dirty data/real_dirty_labeled --train-dir data/dirty_clean --val-dir data/real_dirty_val --val-ratio 0.2 --seed 42

# 4d. baseline ДО дообучения
python -m scripts.eval_real_dirty --val-dir data/real_dirty_val --checkpoint weights/multi_task_best.ckpt

# 4e. регенерация синтетики (с mud_crust) + дообучение с тёплого старта
python -m src.data.generate_dirty_dataset --clean-dir data/processed --output-dir data/dirty_clean --dirty-per-clean 2 --seed 42
python -m src.training.multi_task_train --data-dir data/dirty_clean --init-checkpoint weights/multi_task_best.ckpt --num-unfrozen-stages 1 --lr 1e-4 --max-epochs 8 --early-stopping-patience 3 --weights-dir weights/ft_real_dirty_vN --accelerator gpu
```

Полный прогон (без `--init-checkpoint`) — только при смене
архитектуры или катастрофическом расхождении данных.

```powershell
# 4f. оценка ПОСЛЕ (код возврата 0 при recall >= 0.90, иначе 2)
python -m scripts.eval_real_dirty --val-dir data/real_dirty_val --checkpoint weights/multi_task_best.ckpt --target-recall 0.90
```

Реальные грязные фото уходят в `data/dirty_clean/train/<class>/dirty/` с
префиксом `real_`. Held-out `data/real_dirty_val` в обучении не участвует.
Свежий, ни разу не виденный резерв реальной грязи —
`data/real_dirty_reserve` (см. `docs/DATA.md`).

> **Следующая итерация данных.** После чистки наборов от снега и мусора
> `data/real_dirty_val` сжался до 28 фото, dirty recall = 0.893 (3 трудных
> промаха, отставание от цели 0.90 в пределах шума). План: собрать 20-30 новых
> реальных грязных фото (депо / веб) — восстановить знаменатель val и добавить
> в train 3 трудных кейса для дообучения с тёплого старта (фаза 2, «Дообучение
> с тёплого старта»).

## Триггер переобучения из фидбека

Отдельный вход в цикл, когда исправления пользователей уже накоплены эндпоинтом
`/feedback` в `data/feedback/<family>/` (структура пула — `docs/DATA.md`).

1. **Накопление.** Дождаться порога в `data/feedback` — ориентир **50** новых
   примеров (меньше — переобучение почти не двигает метрики):

   ```powershell
   (Get-ChildItem data/feedback -Recurse -Include *.jpg,*.jpeg,*.png).Count
   ```

2. **Сухой прогон.** Посмотреть, сколько пройдёт валидацию и дедупликацию:

   ```powershell
   python -m scripts.ingest_feedback --require-user-state --verbose
   ```

3. **Вливание.** Скопировать валидные фото в обучающую выборку с префиксом
   `feedback_`. В строгом режиме кадры без `user_state` остаются в очереди.
   Точные совпадения с `real_dirty_val` и `real_clean_probe` пропускаются.
   Для других контрольных наборов укажите `--held-out-dir <путь>`.

   ```powershell
   python -m scripts.ingest_feedback --require-user-state --apply --verbose
   ```

   Результат: `data/dirty_clean/train/<family>/<state>/feedback_*`. Код возврата
   `1`, если были ошибки валидации, — разобрать их перед обучением.

4. **Регенерация синтетики** — только если нужен свежий `mud_crust` поверх
   обновлённого `data/processed` (если чистое дерево не менялось, шаг можно
   пропустить):

   ```powershell
   python -m src.data.generate_dirty_dataset --clean-dir data/processed --output-dir data/dirty_clean --dirty-per-clean 2 --seed 42
   ```

5. **Переобучение.** При небольшой feedback-добавке предпочтителен тёплый старт
   с рабочего чекпоинта (`--init-checkpoint`, пониженный `--lr`, короткий
   `--max-epochs` с `--early-stopping-patience 3..4` — см. «Дообучение с тёплого
   старта» в фазе 2), иначе — полный прогон по фазам 2–3. Затем оценка на
   реальной грязи по фазе 4f (`eval_real_dirty`, цель dirty recall ≥ 0.90) и,
   если фидбэк был про состояние, `scripts/error_analysis_state.py --sweep` для
   перекалибровки `api.state_dirty_threshold`. Лучший чекпоинт сделать рабочим,
   обновить порог и метрики в `configs/config.yaml`, `reports/MODEL_CARD.md` и `docs/PROJECT_CONTEXT.md`.

## Фаза 5. Проверки перед выкладкой

```powershell
python -c "from src.config.config_loader import load_config; assert load_config().image_size == 384; print('image_size: 384')"
pre-commit run --all-files
pytest tests/ -q

docker build -t tractor-vision:cpu .
docker compose up --build -d api
curl http://localhost:8000/health
curl http://localhost:8000/models   # accuracy из метаданных чекпоинта
```
