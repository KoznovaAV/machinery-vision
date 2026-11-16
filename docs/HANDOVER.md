# Развёртывание и передача

Комплект состоит из репозитория GitHub и архива `machinery-vision-resources.zip`.

| Источник | Содержание |
|---|---|
| GitHub | Код, справочник классов, обученные классификаторы и пороги, тесты, документация, отчёты |
| Архив ресурсов | `datasets/`, `weights/`, `offline/`, `test-photos/`, манифест SHA-256 |

Архив не содержит копию исходного кода, `.venv`, пользовательские загрузки и кэши. Его нужно распаковать непосредственно в корень клонированного репозитория. После распаковки `datasets/` и `weights/` находятся рядом с `src/` и `scripts/`.

## Установка с интернетом

Установите Python 3.11 и зависимости по [README](../README.md). Базовые веса можно получить через `python -m scripts.setup_weights`. Датасет передаётся в архиве ресурсов.

## Офлайн-установка Windows x64

Все команды выполняются из корня клона после распаковки архива. Каталог может находиться на любом диске.

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONNOUSERSITE = '1'
$env:TEMP = Join-Path $PWD '.cache/tmp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force $env:TEMP | Out-Null
& '.\offline\runtime\cpython-3.11.15-windows-x86_64-none\python.exe' -B -m venv .venv
& '.\.venv\Scripts\python.exe' -B -m pip --isolated --disable-pip-version-check install --no-index --no-cache-dir --find-links .\offline\environment\wheels -r .\offline\environment\requirements-windows.lock.txt
& '.\.venv\Scripts\python.exe' -B -m scripts.verify_resources
& '.\.venv\Scripts\python.exe' -B -m scripts.check_release
& '.\.venv\Scripts\python.exe' -B -m pip check
& '.\.venv\Scripts\python.exe' -B -m scripts.web
```

Адрес локального интерфейса: http://127.0.0.1:7864/ . Остановка — `Ctrl+C`. Для Linux/macOS используются Python и зависимости соответствующей платформы; вложенные Windows wheels не подходят.

## Данные, проверка и обучение

```bash
python -m scripts.verify_resources
python -m scripts.audit_data
python -m scripts.benchmark
python -m scripts.evaluate
python -m pytest tests -q
python -m scripts.train --task mtz
```

Команды предполагают активированное окружение. Другие задачи обучения: `families`, `conditions`, `regions`. Кандидаты сохраняются в `runs/training/`; установка в рабочий комплект выполняется отдельно. После изменения датасета исходный манифест ресурсов перестаёт соответствовать файлам: для нового комплекта нужен новый манифест.

Тестовые изображения находятся в `test-photos/`. В интерфейсе выбирается отдельный JPG/PNG; `index.csv` содержит контрольные метки и фактические ответы. Набор включает известные сложные случаи и не является независимой оценкой точности.

`artifacts/mvp/current_release.json` определяет рабочий комплект. Для отката сохраняются согласованные версии кода, справочника, классификаторов и порогов. Возможности и ограничения: [карточка модели](MODEL_CARD.md). Правила разметки: [данные](DATA.md).

## Временный публичный доступ

Запустите отдельный экземпляр:

```bash
python -m scripts.web --public-demo --port 7866
```

В другом терминале из той же папки:

```bash
ssh -T -o UserKnownHostsFile=.cache/demo_known_hosts -o IdentityFile=none -o IdentitiesOnly=yes -o ServerAliveInterval=30 -R 80:127.0.0.1:7866 nokey@localhost.run
```

OpenSSH выдаёт публичный HTTPS-адрес. Оба процесса должны оставаться работающими. Бесплатные адреса меняются; [условия провайдера](https://localhost.run/docs/forever-free).

Публичный режим хранит загрузки отдельно в `runs/public-demo/`, удаляет метаданные изображений и закрывает JSON/HTML артефакты и документацию API. Изображения доступны 20 минут. Очистка работает при запущенном сервере и возобновляется после перезапуска. Режим рассчитан на один worker-процесс. Постоянное размещение требует отдельной настройки доступа, нагрузки и хранения.
