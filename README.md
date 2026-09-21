# Система NER на YandexGPT

Извлечение именованных сущностей (Named Entity Recognition) из юридических документов с использованием YandexGPT.

## Описание

Этот проект реализует пайплайн для автоматического извлечения структурированных данных из юридических документов:
- Реквизиты сторон (названия компаний, ИНН, КПП)
- Даты (подписание, сроки действия)
- Суммы (финансовые значения с валютой)
- Сроки обязательств (дни, месяцы, годы)

## Структура проекта

```
hw3/
├── .env                      # Переменные окружения (API ключи)
├── .gitignore               # Игнорирование чувствительных файлов
├── HW3_YandexGPT_NER.ipynb  # Jupyter Notebook с отчетом
└── src/
   └── ner/
       ├── __init__.py
       ├── dataset_utils.py      # Работа с датасетом
       ├── prompts.py            # Промпты для NER
       ├── yandexgpt_client.py   # API клиент
       ├── parser.py             # Парсер ответов
       ├── chunking.py           # Разбиение документов
       ├── extractor.py          # Основной экстрактор
       └── test_ner.py           # Скрипт тестирования

```

## Установка

1. **Установите Python** (версия 3.10+)

2. **Установите зависимости**:
```bash
pip install requests python-dotenv datasets pandas
```

3. **Настройте API ключи**:
```bash
# Создайте .env файл (скопируйте из .env.example)
cp .env.example .env
```

4. **Получите ключи Yandex Cloud**:
   - перейдите в [Yandex Cloud Console](https://console.cloud.yandex.ru)
   - Создайте сервисный аккаунт с ролью `ai.languageModels.user`
   - Получите API-ключ и идентификатор каталога
   - Добавьте в `.env`:
   ```
   YC_API_KEY=ваш_api_ключ
   YC_FOLDER_ID=ваш_folder_id
   ```

## Использование

### Быстрый старт

```python
from src.ner.dataset_utils import LegalDataset
from src.ner.extractor import NERExtractor

# Инициализация экстрактора (ключи из .env)
extractor = NERExtractor()

# Загрузка датасета
dataset = LegalDataset()
dataset.load("train")

# Извлечение сущностей из документа
sample = dataset.get_sample("train", index=0)
result = extractor.extract_from_text(sample['text'])

# Просмотр результатов
print(f"Компании: {result.metrics['total_companies']}")
print(f"Суммы: {result.metrics['total_amounts']}")
print(f"Сроки: {result.metrics['total_terms']}")
```

### Тестирование

```bash
# Запуск тестов
python -m src.ner.test_ner
```

### Jupyter Notebook

Откройте `HW3_YandexGPT_NER.ipynb` для интерактивного анализа и примеров.

## Промпты

### Системный промпт

Модель настраивается как AI-помощник юриста с задачами:
- Извлекать факты без искажений
- Возвращать только валидный JSON
- Не выдумывать данные (использовать null при отсутствии)
- Учитывать контекст документа

### Структура ответа JSON

```json
{
  "companies": [
    {"name": "ООО \"Вектор\"", "inn": "7701234567", "kpp": "770101001"}
  ],
  "dates": [
    {"description": "Дата договора", "value": "2023-03-15"}
  ],
  "amounts": [
    {"description": "Стоимость услуг", "value": 500000, "currency": "RUB"}
  ],
  "terms": [
    {"description": "Срок оплаты", "value": 10, "unit": "days"}
  ]
}
```

## Датасеты

Используется датасет [russian-legal-ner](https://huggingface.co/datasets/TryDotAtwo/russian-legal-ner):
- 97.7k строк юридических документов
- Русский язык
- Задача: Token Classification (NER)

## Edge Cases

Проект включает тестирование для:
1. **Отсутствие данных** - документы без сумм/срочков
2. **Длинные документы** - разбиение на чанки с перекрытием
3. **Ошибка JSON** - обработка невалидных ответов
4. **Зашумленный текст** - опечатки, OCR-ошибки

### Когда использовать:
- **LLM (YandexGPT)**: редко встречающиеся паттерны, сложные контексты, быстрая настройка
- **Классические ML (Natasha/Spacy)**: массовая обработка, low-cost, стабильность

## Тарифы YandexGPT

Информация: https://cloud.yandex.ru/docs/yandexgpt/pricing

## Безопасность

- API ключи хранятся в `.env` и игнорируются git