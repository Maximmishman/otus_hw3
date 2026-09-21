"""Основной модуль для извлечения сущностей из юридических документов."""

import random
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from .dataset_utils import LegalDataset
from .yandexgpt_client import YandexGPTClient
from .prompts import SYSTEM_PROMPT
from .parser import NERParser
from .chunking import DocumentChunker


@dataclass
class ExtractionResult:
    """Результат извлечения сущностей."""
    document_id: str
    original_text: str
    extracted_data: Dict[str, Any]
    success: bool
    error_message: Optional[str]
    metrics: Dict[str, Any]
    chunks_used: int = 1


class NERExtractor:
    """Экстрактор сущностей из юридических документов."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        folder_id: Optional[str] = None,
        max_tokens_per_chunk: int = 4000,
        overlap_ratio: float = 0.1,
        temperature: float = 0.2
    ):
        self.temperature = temperature
        """
        Инициализация экстрактора.
        
        Args:
            api_key: API-ключ Yandex Cloud
            folder_id: Идентификатор каталога Yandex Cloud
            max_tokens_per_chunk: Максимальное количество токенов в чанке
            overlap_ratio: Доля перекрытия чанков
        """
        self.client = YandexGPTClient(api_key=api_key, folder_id=folder_id)
        self.parser = NERParser()
        self.chunker = DocumentChunker(
            max_tokens=max_tokens_per_chunk,
            overlap_ratio=overlap_ratio
        )
        
    def extract_from_text(
        self,
        text: str,
        document_id: str = "doc_001",
        temperature: float = 0.2
    ) -> ExtractionResult:
        """
        Извлечь сущности из текста документа.
        
        Args:
            text: Текст документа
            document_id: ID документа для трасировки
            temperature: Температура генерации
            
        Returns:
            Результат извлечения
        """
        # Разбиение на чанки если нужно
        chunks = self.chunker.chunk(text)
        chunks_used = len(chunks)
        
        # Если документ длинный, объединяем чанки
        if chunks_used > 1:
            # Обработка по чанкам (можно объединить результаты)
            all_results = []
            for chunk in chunks:
                response, _ = self.client.generate(
                    SYSTEM_PROMPT,
                    chunk.text,
                    temperature=temperature
                )
                parsed = self.parser.extract_entities(response)
                all_results.append(parsed)
            
            # Объединение результатов (упрощенно)
            extracted_data = {
                "companies": [],
                "dates": [],
                "amounts": [],
                "terms": []
            }
            for res in all_results:
                for key in extracted_data:
                    extracted_data[key].extend(res.get(key, []))
                    
            metrics = self.parser.calculate_metrics(extracted_data)
            return ExtractionResult(
                document_id=document_id,
                original_text=text,
                extracted_data=extracted_data,
                success=True,
                error_message=None,
                metrics=metrics,
                chunks_used=chunks_used
            )
        else:
            # Один чанк - простая обработка
            text_to_analyze = text
            if len(text) > 10000:  # Слишком длинный документ
                text_to_analyze = text[:10000] + "\n...(текст обрезан)"
                
            response, _ = self.client.generate(
                SYSTEM_PROMPT,
                text_to_analyze,
                temperature=self.temperature
            )
            
            extracted_data = self.parser.extract_entities(response)
            metrics = self.parser.calculate_metrics(extracted_data)
            
            return ExtractionResult(
                document_id=document_id,
                original_text=text,
                extracted_data=extracted_data,
                success=extracted_data.get("error") is None,
                error_message=extracted_data.get("error"),
                metrics=metrics,
                chunks_used=1
            )
    
    def extract_from_dataset(
        self,
        dataset: LegalDataset,
        n_samples: int = 10,
        split: str = "test",
        temperature: float = 0.2,
        save_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Извлечь сущности из выборки датасета.
        
        Args:
            dataset: Объект датасета
            n_samples: Количество документов
            split: Часть датасета
            temperature: Температура
            save_path: Путь для сохранения результатов (опционально)
            
        Returns:
            Список результатов
        """
        if split == "test":
            samples = dataset.test_data
        else:
            samples = dataset.train_data
            
        results = []
        
        for i in range(min(n_samples, len(samples))):
            doc = samples[i]
            result = self.extract_from_text(
                doc["text"],
                document_id=f"{split}_{i:03d}",
                temperature=temperature
            )
            results.append({
                "document_id": result.document_id,
                "text_preview": result.original_text[:200] + "..." if len(result.original_text) > 200 else result.original_text,
                "extracted": result.extracted_data,
                "success": result.success,
                "metrics": result.metrics,
                "chunks_used": result.chunks_used
            })
            
        # Сохранение результатов
        if save_path:
            self._save_results(results, save_path)
            
        return results
    
    def extract_specific_cases(
        self,
        dataset: LegalDataset,
        n_no_amount: int = 3,
        temperature: float = 0.2
    ) -> List[Dict[str, Any]]:
        """
        Тестирование edge cases.
        
        Args:
            dataset: Объект датасета
            n_no_amount: Количество документов без сумм
            temperature: Температура
            
        Returns:
            Результаты тестирования
        """
        results = []
        
        # Тест 1: Документы без сумм
        print("\nТест 1: Документы без сумм (проверка на галлюцинации)...")
        docs_without_amount = dataset.get_documents_without_amount(n_no_amount)
        
        for i, doc_text in enumerate(docs_without_amount):
            result = self.extract_from_text(
                doc_text,
                document_id=f"edge_case_no_amount_{i:03d}",
                temperature=temperature
            )
            
            # Проверяем, нет ли галлюцинаций (сумм там не должно быть)
            has_hallucination = len(result.extracted_data.get("amounts", [])) > 0
            
            results.append({
                "case_type": "no_amount",
                "document_id": result.document_id,
                "success": result.success,
                "has_hallucination": has_hallucination,
                "metrics": result.metrics
            })
            print(f"  Документ {i+1}: галлюцинаций = {has_hallucination}")
            
        return results
    
    def _save_results(self, results: List[Dict[str, Any]], path: str) -> None:
        """Сохранить результаты в JSON файл."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Результаты сохранены в {path}")
        
    def save_single_result(self, result: ExtractionResult, path: str) -> None:
        """Сохранить одиночный результат."""
        data = {
            "document_id": result.document_id,
            "chunks_used": result.chunks_used,
            "success": result.success,
            "error": result.error_message,
            "extracted": result.extracted_data,
            "metrics": result.metrics
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
