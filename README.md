# Обнаружение финансовых мошенничеств с помощью машинного обучения

Этот репозиторий содержит код для обучения и реализации моделей, предназначенных для выявления подозрительных карточных операций и предотвращения мошенничества.

## Данные
Используются анонимизированные данные карточных транзакций. В них содержатся признаки, позволяющие определить является ли транзакция мошеннической.

## Алгоритмы
Реализованы две модели:
- Логистическая регрессия
- Глубовая нейронная сеть 
- Сверточная Нейронная сеть 
## Обучение
Производится предварительная обработка данных, сбалансировка классов и разделение на тренировочную/тестовую выборки. Затем обучаются и оцениваются модели.

## Результаты
Сравниваются метрики качества обеих моделей по тестовой выборке.

