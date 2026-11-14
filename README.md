# Machinery Vision

<p align="center">
  <img alt="Python 3.11" src="https://img.shields.io/badge/PYTHON-3.11-00ADD8?style=for-the-badge&amp;logo=python&amp;logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PYTORCH-2.x-61DAFB?style=for-the-badge&amp;logo=pytorch&amp;logoColor=white">
  <img alt="Grounding DINO" src="https://img.shields.io/badge/GROUNDING%20DINO-1D9BF0?style=for-the-badge">
  <img alt="FastAPI" src="https://img.shields.io/badge/FASTAPI-FF6B00?style=for-the-badge&amp;logo=fastapi&amp;logoColor=white">
</p>

Компьютерное зрение для распознавания техники и запчастей.

Экспериментальный веб-интерфейс выделяет машины на фото, определяет тип и оценивает семейство. Для МТЗ-80/82/820 отдельный классификатор оценивает группу модификаций. OCR добавляет кандидатов по маркировке.

## Веб-интерфейс иерархического распознавания

[Запуск в PowerShell и подготовка весов](docs/HIERARCHY_WEB.md) · [Метрики этапа 6](artifacts/hierarchy/metrics.json).

На независимом контроле из 158 фото система выдала 107 ответов о семействе, 103 из них верные. Точных моделей автоматически не подтвердила. Веб-интерфейс работает локально.

## Классы и данные

- В модели 8 семейств-кандидатов. Для 5 есть пороги выдачи ответа о семействе.
- В независимом контроле 158 фото. Обучение использовало 431 фото с рабочими метками и 5 с меткой только семейства.
- [Справочник классов](configs/machinery_taxonomy.json) включает группы модификаций, серии и семейства.

Репозиторий содержит код, обученные головы с порогами и сводные метрики. Исходные фотографии и рабочие датасеты размещены в локальной копии проекта. Инструкция по веб-интерфейсу включает загрузку базовых весов из открытых источников.
