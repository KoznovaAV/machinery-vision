# Локальный веб-интерфейс

[Текущий локальный выпуск и ограничения](CURRENT.md). Используйте порт 7864.

## Запуск в рабочей копии на D

В PowerShell выполните:

```powershell
Set-Location 'D:/machinery-vision/project'
$python = 'D:/machinery-vision/envs/tractor/Scripts/python.exe'
$env:TORCH_HOME = 'D:/machinery-vision/envs/torch-cache'
$env:TEMP = "$PWD/output"
$env:TMP = $env:TEMP
$env:NO_ALBUMENTATIONS_UPDATE = '1'
& $python -B -m scripts.serve_hierarchy_review --port 7864
```

Откройте <http://127.0.0.1:7864> и загрузите JPEG, PNG или WebP до 8 МБ. Вывод содержит выделенные машины, семейство и кандидатов. При недостатке признаков интерфейс сообщает о неопределённости. Результаты сохраняются в `output/hierarchy_review/`.

В текущем MVP используются отдельные головы состояния из `artifacts/mvp/condition_bundle.json`. Оценка может отсутствовать из-за недостаточной уверенности, снега, плохого освещения, частичного вида или спорных границ объекта. Снег означает снег в кадре; дождь и влажность поверхности пока недостаточно подтверждены. Серия/семейство не выдаются за точную модель.

Прицеп и цистерна не должны считаться отдельными самоходными машинами. Если роль области неясна, она показывается отдельно как неподтверждённая, без оценки её модели и состояния. Перед передачей проверьте целостность комплекта командой `python -B -m scripts.check_mvp_release`.

## Подготовка после нового клонирования

**Новый клон содержит код и компактные артефакты текущего MVP.** Большие базовые энкодеры/детектор нужно загрузить отдельно; фото и рабочие данные обучения остаются в локальной копии. Полные команды с размещением кэшей на D — в [README](../README.md); порядок передачи данных — в [HANDOVER](HANDOVER.md).


Выполните в корне репозитория на нужном диске. Понадобится Python 3.11. Команда подготовки весов загружает около 0,9 ГБ. Она использует опубликованный комплект и сверяет SHA-256 файлов DINOv2, ConvNeXt и Grounding DINO.

```powershell
New-Item -ItemType Directory -Force 'output' | Out-Null
$env:TEMP = "$PWD/output"
$env:TMP = $env:TEMP
$env:TORCH_HOME = "$PWD/weights/torch-cache"
$env:NO_ALBUMENTATIONS_UPDATE = '1'
py -3.11 -m venv .venv
$python = "$PWD/.venv/Scripts/python.exe"
& $python -m pip --no-cache-dir install -r requirements-hierarchy.txt
& $python -B -m scripts.setup_hierarchy_weights
& $python -B -m scripts.check_mvp_release
& $python -B -m scripts.serve_hierarchy_review --port 7864
```

Если окружение уже создано через `uv`, зависимости можно установить командой `uv pip install --python $python -r requirements-hierarchy.txt`. Для этой команды кэш можно задать через `UV_CACHE_DIR`.

Источники весов — [timm DINOv2 Small](https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m), [Grounding DINO Tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny/tree/a2bb814dd30d776dcf7e30523b00659f4f141c71) и [Torchvision ConvNeXt Tiny](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html).

Для ручной проверки в локальной копии используйте `D:/machinery-vision/test-photos/machinery-vision-round6-test-photos.zip`. Файл `index.csv` внутри ZIP содержит эталон, детализацию метки и группу каждого фото. [Опубликованные метрики](../artifacts/hierarchy/metrics.json) относятся к этим 158 контрольным кадрам.
