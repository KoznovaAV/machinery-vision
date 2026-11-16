# Эксперименты

Исторические метрики обучения и параметры экспериментов.

## Обучение: `training_runs/`

Метрики Lightning (`metrics.csv`) и гиперпараметры (`hparams.yaml`) трёх
полных прогонов с ImageNet-инициализации. Значения `val_*` — по эпохам, где
валидация выполнялась.

| Прогон | Дата (mtime) | `num_unfrozen_stages` | Эпох с валидацией | Лучший `val_model_acc` | Лучший `val_state_acc` |
|--------|--------------|-----------------------|-------------------|------------------------|------------------------|
| `version_0` | 2026-08-29 | 2 | 2 | 0.926 (эп. 1) | 0.995 (эп. 1) |
| `version_1` | 2026-08-29 | 1 | 35 | 0.990 (эп. 20) | 0.951 (эп. 32) |
| `version_2` | 2026-08-30 | 1 | 29 | 1.000 (эп. 16) | 0.955 (эп. 22) |

Общие гиперпараметры: `lr 0.0005`, `backbone_lr_factor 0.1`,
`weight_decay 0.0001`, `loss_balancing uncertainty` (`gradnorm_alpha 1.5` не
используется при `uncertainty`).

`version_0` оборван после двух эпох (двухстадийная разморозка);
`version_1` и `version_2` — полные прогоны с одной размороженной стадией.
`val_state_acc` в `version_0` измерен на двух эпохах и несопоставим с остальными.

Параметры и результаты дообучения v2 и v3 описаны в
[model card](../../reports/MODEL_CARD.md) и
[реестре весов](../../reports/weights_inventory.md). Консольный лог v2
сохранён в локальном архиве проекта.

## Оценка состояния: `reports/error_analysis_*`

Три разбора головы состояния (`scripts/error_analysis_state.py`):

| Каталог | Чекпоинт | Наборы (real dirty / synthetic clean / probe) |
|---------|----------|-----------------------------------------------|
| `error_analysis_baseline` | `multi_task_best.ckpt` до дообучения | 32 / 103 / 23 |
| `error_analysis_v2` | `multi_task_best_v2_candidate.ckpt` | 32 / 103 / 23 |
| `error_analysis_v3` | `ft_state_v3/last.ckpt` | 28 / 102 / 19 (после чистки данных) |

Сводка свипа порога `p(dirty)` — [reports/README.md](../../reports/README.md).
