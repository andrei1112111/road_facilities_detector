# Segment

## Описание
Segment — это модель для сегментации дорожной разметки и оценки её качества. Она позволяет выделять разметку на изображениях дорог и анализировать её состояние.
Модель возвращает число `0` если дорожная разметка не найдена, `-n` если разметка выглядит поврежденной и `+n` если на разметке не удалось обнаружить проблемы.

## Возможности
- Сегментация дорожной разметки на изображениях
- Оценка качества разметки
- Поддержка различных форматов входных данных

## Структура проекта
```
Segment/
|── __init__.py
|── utils              # дополнительный функционал
|── segment.py   
│── README.md          # Документация
```

## Авторы
Копылов Матвей НГУ ИИР.
