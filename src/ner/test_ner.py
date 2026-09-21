"""Скрипт для тестирования системы NER."""

import sys
import os

# Добавляем родительскую папку в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ner.dataset_utils import LegalDataset
from src.ner.extractor import NERExtractor


def test_basic_extraction():
    """Базовое тестирование извлечения сущностей."""
    print("=" * 60)
    print("ТЕСТ 1: Базовое извлечение сущностей")
    print("=" * 60)
    
    # Инициализация экстрактора
    extractor = NERExtractor(
        api_key=None,  # Возьмет из .env
        folder_id=None,
        temperature=0.2
    )
    
    # Загрузка датасета
    print("\nЗагрузка датасета...")
    dataset = LegalDataset()
    dataset.load("train")
    
    # Пример 1: Документ с полной информацией
    print("\nПример 1: Документ с полной информацией")
    sample = dataset.get_sample("train")
    print(f"Текст документа (первые 300 символов):\n{sample['text'][:300]}...\n")
    
    result = extractor.extract_from_text(
        sample["text"],
        document_id="test_basic_001",
        temperature=0.2
    )
    
    print(f"\nРезультат извлечения:")
    print(f"  Компании: {len(result.extracted_data.get('companies', []))}")
    print(f"  Даты: {len(result.extracted_data.get('dates', []))}")
    print(f"  Суммы: {len(result.extracted_data.get('amounts', []))}")
    print(f"  Сроки: {len(result.extracted_data.get('terms', []))}")
    print(f"  Успех: {result.success}")
    if result.error_message:
        print(f"  Ошибка: {result.error_message}")
        
    if result.extracted_data.get("companies"):
        print("\nИзвлеченные компании:")
        for comp in result.extracted_data["companies"][:3]:
            print(f"  - {comp.get('name', 'N/A')} (ИНН: {comp.get('inn', 'N/A')})")
    
    if result.extracted_data.get("amounts"):
        print("\nИзвлеченные суммы:")
        for amt in result.extracted_data["amounts"][:3]:
            print(f"  - {amt.get('description', 'N/A')}: {amt.get('value', 0)} {amt.get('currency', 'RUB')}")
    
    return result


def test_edge_cases():
    """Тестирование edge cases."""
    print("\n" + "=" * 60)
    print("ТЕСТ 2: Edge Cases")
    print("=" * 60)
    
    extractor = NERExtractor()
    dataset = LegalDataset()
    dataset.load("train")
    
    # Тест на отсутствие данных
    print("\nПроверка документов без сумм (галлюцинации)...")
    results = extractor.extract_specific_cases(dataset, n_no_amount=5)
    
    total_docs = len(results)
    hallucinations = sum(1 for r in results if r.get("has_hallucination"))
    
    print(f"\nРезультаты edge case тестов:")
    print(f"  Всего документов: {total_docs}")
    print(f"  С галлюцинациями: {hallucinations}")
    print(f"  Без галлюцинаций: {total_docs - hallucinations}")
    print(f"  Процент корректности: {(1 - hallucinations/total_docs) * 100:.1f}%")
    
    return results


def test_long_documents():
    """Тестирование длинных документов."""
    print("\n" + "=" * 60)
    print("ТЕСТ 3: Длинные документы")
    print("=" * 60)
    
    extractor = NERExtractor()
    dataset = LegalDataset()
    dataset.load("train")
    
    # Поиск длинных документов
    print("\nПоиск длинных документов (более 5000 символов)...")
    samples = dataset.get_random_samples("train", 10)
    
    long_doc_count = sum(1 for s in samples if len(s["text"]) > 5000)
    print(f"Длинных документов: {long_doc_count} из 10")
    
    if long_doc_count > 0:
        # Берем первый длинный документ
        for sample in samples:
            if len(sample["text"]) > 5000:
                print(f"\nОбработка длинного документа ({len(sample['text'])} символов)...")
                result = extractor.extract_from_text(
                    sample["text"],
                    document_id="test_long_001",
                    temperature=0.2
                )
                
                print(f"\nРезультат:")
                print(f"  Использовано чанков: {result.chunks_used}")
                print(f"  Компании: {len(result.extracted_data.get('companies', []))}")
                print(f"  Суммы: {len(result.extracted_data.get('amounts', []))}")
                print(f"  Успех: {result.success}")
                break
    
    return True


def test_multiple_documents():
    """Обработка выборки документов."""
    print("\n" + "=" * 60)
    print("ТЕСТ 4: Обработка выборки документов")
    print("=" * 60)
    
    extractor = NERExtractor()
    dataset = LegalDataset()
    dataset.load("test")
    
    # Тестирование на тестовой выборке
    print("\nТестирование на 7 документах из test выборки...")
    n_docs = 7
    results = extractor.extract_from_dataset(dataset, n_samples=n_docs, split="test")
    
    # Статистика
    successful = sum(1 for r in results if r["success"])
    total_companies = sum(r["metrics"]["total_companies"] for r in results)
    total_amounts = sum(r["metrics"]["total_amounts"] for r in results)
    
    print(f"\nСтатистика по выборке ({n_docs} документов):")
    print(f"  Успешных: {successful}/{n_docs}")
    print(f"  Всего компаний: {total_companies}")
    print(f"  Всего сумм: {total_amounts}")
    print(f"  Успешность: {successful/n_docs * 100:.1f}%")
    
    return results


def main():
    """Основная функция тестирования."""
    print("=" * 60)
    print("Тестирование NER системы на YandexGPT")
    print("=" * 60)
    
    try:
        # Базовое тестирование
        basic_result = test_basic_extraction()
        
        # Edge cases
        edge_results = test_edge_cases()
        
        # Длинные документы
        test_long_documents()
        
        # Выборка документов
        batch_results = test_multiple_documents()
        
        print("\n" + "=" * 60)
        print("ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ УСПЕШНО")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nОШИБКА при тестировании: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
