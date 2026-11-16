# Реестр весов

В чистой локальной копии используются `multi_task_best_v3.ckpt` (API)
и `multi_task_best.ckpt` (v2, скрипты по умолчанию). Таблица ниже фиксирует
исходный архив весов, включая исторические имена и совпадения по SHA-256.

Файлы из `weights/` (вне git). Размер, время изменения и SHA-256 сняты с рабочей машины; одинаковый SHA-256 означает побайтово идентичные файлы. Роли выведены из совпадения хэшей, `docs/RETRAIN.md`, `reports/MODEL_CARD.md` и путей в `reports/*/report.json`.

| Файл | МБ | Изменён | SHA-256 | Роль |
|------|----|---------|---------|------|
| `multi-task-best-epoch=13-val_model_acc=1.000.ckpt` | 224.3 | 2026-08-30 20:51 | `11e1f6ac0cd797a517573f09fdebd41255ccbc42c0b94ebf19e9c6f2231f4d7f` | v1: лучший чекпоинт полного обучения; источник для `--init-checkpoint` при дообучении v2 (идентичен `multi_task_best_before_v2.ckpt`) |
| `multi_task_best.ckpt` | 224.3 | 2026-08-31 17:24 | `3c1e55414856e9107b9d3e15807d3704c6cc4095248b2536dc997c3c6a33e38b` | рабочий файл `configs/config.yaml:weights.multi_task` = fine-tune v2 (идентичен `multi_task_best_v2_candidate.ckpt`) |
| `multi_task_best_before_v2.ckpt` | 224.3 | 2026-08-30 20:51 | `11e1f6ac0cd797a517573f09fdebd41255ccbc42c0b94ebf19e9c6f2231f4d7f` | резервная копия рабочего файла перед промоутом v2 (идентична `multi-task-best-epoch=13-...`) |
| `multi_task_best_v1.ckpt` | 224.3 | 2026-08-29 16:18 | `22a3d381fbe8eaf49e1e3cd0deffa529179903cf07f474dbacbf3eedc54dfe00` | ранний чекпоинт v1 от 2026-08-29 |
| `multi_task_best_v2_candidate.ckpt` | 224.3 | 2026-08-31 17:15 | `3c1e55414856e9107b9d3e15807d3704c6cc4095248b2536dc997c3c6a33e38b` | архивная копия v2 (идентична `multi_task_best.ckpt`) |
| `multi_task_best_v3.ckpt` | 224.3 | 2026-09-11 11:57 | `78964cf5dd564f3bb6673677d1f9e55ac9c6479304b83170c4a801c205c7bb03` | v3: чекпоинт реестра `configs/config.yaml:models.machine` (идентичен `ft_state_v3/last.ckpt`) |
| `ft_state_v2/last.ckpt` | 224.3 | 2026-08-31 17:12 | `3c1e55414856e9107b9d3e15807d3704c6cc4095248b2536dc997c3c6a33e38b` | выход прогона дообучения v2: последняя эпоха (идентичен `multi_task_best.ckpt`) |
| `ft_state_v2/multi-task-best-epoch=02-val_model_acc=0.997.ckpt` | 224.3 | 2026-08-31 17:12 | `3c1e55414856e9107b9d3e15807d3704c6cc4095248b2536dc997c3c6a33e38b` | выход прогона дообучения v2: лучшая эпоха 2, `val_model_acc` 0.997 (идентичен `multi_task_best.ckpt`) |
| `ft_state_v2/multi_task_final.ckpt` | 224.3 | 2026-08-31 17:15 | `5518046e3cbf3878ee1cdc0a89c6e813b269a77b1466e8fe624d4973663920d7` | выход прогона дообучения v2: финальные веса |
| `ft_state_v2/train.log` | 0.4 | 2026-08-31 17:15 | `69f5a0cb39889ed91e1477bb7aaac7e02bb5d976dbfaf65f79bb30a1814230d1` | лог прогона дообучения v2 (скопирован в `docs/experiments/training_runs/ft_state_v2_train.log`) |
| `ft_state_v3/last.ckpt` | 224.3 | 2026-09-11 11:57 | `78964cf5dd564f3bb6673677d1f9e55ac9c6479304b83170c4a801c205c7bb03` | выход прогона v3: последняя эпоха; чекпоинт разбора ошибок `reports/error_analysis_v3` (идентичен `multi_task_best_v3.ckpt`) |
| `ft_state_v3/multi-task-best-epoch=00-val_model_acc=1.000.ckpt` | 224.3 | 2026-09-11 11:56 | `208b5d79f8698405af8c3e31b41d6de737d642dcd2578d4169595ed4ae6b15e7` | выход прогона v3: лучшая по `val_model_acc` эпоха 0 |
| `ft_state_v3/multi_task_final.ckpt` | 224.3 | 2026-09-11 11:59 | `af7a51554acd0d90265aa19a679048e3c5e86ae1094a35335a04511c960bfa1c` | выход прогона v3: финальные веса |

## Идентичные группы

- `3c1e55414856…` — `multi_task_best.ckpt`, `multi_task_best_v2_candidate.ckpt`, `ft_state_v2/last.ckpt`, `ft_state_v2/multi-task-best-epoch=02-val_model_acc=0.997.ckpt` (v2).
- `11e1f6ac0cd7…` — `multi-task-best-epoch=13-val_model_acc=1.000.ckpt`, `multi_task_best_before_v2.ckpt` (v1, источник дообучения v2).
- `78964cf5dd56…` — `multi_task_best_v3.ckpt`, `ft_state_v3/last.ckpt` (v3).

Дубликаты по 224 МБ можно не хранить: достаточно по одному файлу на SHA-256.

## Проверка и метаданные

SHA-256 всех 13 файлов `weights/` пересчитаны 2026-10-22 командой `Get-FileHash` и
сверены с таблицей выше: 13 из 13 совпали.

Метаданные из самих чекпоинтов (`torch.load`, ключи `epoch`, `global_step`,
`hyper_parameters`, `callbacks`); все обучены на PyTorch Lightning 2.6.5,
`num_unfrozen_stages=1`, `backbone_lr_factor=0.1`, `weight_decay=1e-4`,
`loss_balancing=uncertainty`.

| Файл | Эпоха | Шаг | LR | `best_model_score` (`val_model_acc`) |
|------|-------|-----|----|-------------------------------------|
| `multi_task_best_v1.ckpt` | 19 | 3560 | 5e-4 | 0.990 |
| `multi-task-best-epoch=13-val_model_acc=1.000.ckpt`, `multi_task_best_before_v2.ckpt` | 13 | 2898 | 5e-4 | 1.000 |
| `multi_task_best.ckpt`, `multi_task_best_v2_candidate.ckpt`, `ft_state_v2/last.ckpt`, `ft_state_v2/multi-task-best-epoch=02-…ckpt` | 2 | 633 | 1e-4 | 0.997 |
| `ft_state_v2/multi_task_final.ckpt` | 6 | 1266 | 1e-4 | 0.997 |
| `multi_task_best_v3.ckpt`, `ft_state_v3/last.ckpt`, `ft_state_v3/multi-task-best-epoch=00-…ckpt` | 0 | 205 | 1e-4 | 1.000 |
| `ft_state_v3/multi_task_final.ckpt` | 4 | 820 | 1e-4 | 1.000 |

`ft_state_v3/last.ckpt` и файл лучшей эпохи v3 имеют одинаковые эпоху и шаг, но разные
SHA-256 (различается, по-видимому, состояние колбэков; причина не проверялась).
Метрики состояния и реальной грязи для каждого чекпоинта — `recheck_2026-10-22.md`
(v1-эпоха-13, v2, v3) и `error_analysis_*/` (v2, v3, baseline).
