"""Модуль для парсинга ответов от YandexGPT."""

import re
import json
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime


class NERParser:
    """Парсер ответов модели для извлечения NER-сущностей."""
    
    def __init__(self):
        self.expected_schema = {
            "companies": list,
            "dates": list,
            "amounts": list,
            "terms": list
        }
        
    def clean_response(self, response: str) -> str:
        """
        Очистить ответ от Markdown-тегов и избыточного текста.
        
        Args:
            response: Сырой ответ от модели
            
        Returns:
            Очищенная строка
        """
        # Удаление markdown кода ```json и ```
        cleaned = re.sub(r'```json\s*', '', response, flags=re.IGNORECASE)
        cleaned = re.sub(r'```', '', cleaned, flags=re.IGNORECASE)
        
        # Удаление лишнего текста до/после JSON
        # Ищем первое '{' и последнее '}'
        start_idx = cleaned.find('{')
        end_idx = cleaned.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            cleaned = cleaned[start_idx:end_idx + 1]
            
        return cleaned.strip()
    
    def parse_json_response(self, response: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
       Parse JSON ответ.
        
        Args:
            response: Очищенный ответ (должен быть JSON)
            
        Returns:
            Кортеж (parsed_data, error_message)
        """
        try:
            cleaned = self.clean_response(response)
            data = json.loads(cleaned)
            return data, None
        except json.JSONDecodeError as e:
            return None, f"JSON Decode Error: {str(e)}"
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def validate_schema(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Валидировать структуру данных.
        
        Args:
            data: Парсер данные
            
        Returns:
            Кортеж (is_valid, errors)
        """
        errors = []
        
        for key, expected_type in self.expected_schema.items():
            if key not in data:
                errors.append(f"Отсутствует поле: {key}")
            elif not isinstance(data[key], expected_type):
                errors.append(f"Неверный тип для {key}: ожидаем {expected_type.__name__}")
                
        return len(errors) == 0, errors
    
    def extract_entities(self, response: str) -> Dict[str, Any]:
        """
        Извлечь сущности из ответа модели.
        
        Args:
            response: Ответ от модели
            
        Returns:
            Словарь с извлеченными сущностями
        """
        # Парсинг JSON
        data, error = self.parse_json_response(response)
        
        if error or data is None:
            # Возвращаем структуру с null при ошибке
            return {
                "error": error or "Unknown error",
                "companies": [],
                "dates": [],
                "amounts": [],
                "terms": []
            }
        
        # Валидация
        is_valid, validation_errors = self.validate_schema(data)
        
        if not is_valid:
            return {
                "validation_errors": validation_errors,
                "partial_data": data,
                "companies": data.get("companies", []),
                "dates": data.get("dates", []),
                "amounts": data.get("amounts", []),
                "terms": data.get("terms", [])
            }
        
        return {
            "error": None,
            "companies": data.get("companies", []),
            "dates": data.get("dates", []),
            "amounts": data.get("amounts", []),
            "terms": data.get("terms", [])
        }
    
    def calculate_metrics(
        self,
        extracted: Dict[str, Any],
        ground_truth: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Рассчитать метрики качества извлечения.
        
        Args:
            extracted: Извлеченные данные
            ground_truth:Ground truth для сравнения (опционально)
            
        Returns:
            Словарь с метриками
        """
        metrics = {
            "total_companies": len(extracted.get("companies", [])),
            "total_dates": len(extracted.get("dates", [])),
            "total_amounts": len(extracted.get("amounts", [])),
            "total_terms": len(extracted.get("terms", [])),
            "has_error": extracted.get("error") is not None,
            "error_message": extracted.get("error")
        }
        
        # Подсчет сумм
        total_amount = sum(
            amount.get("value", 0) or 0
            for amount in extracted.get("amounts", [])
        )
        metrics["total_amount"] = total_amount
        
        # Если есть ground truth, можно сравнить
        if ground_truth:
            metrics["comparison"] = {
                "companies_gt": len(ground_truth.get("companies", [])),
                "dates_gt": len(ground_truth.get("dates", [])),
                "amounts_gt": len(ground_truth.get("amounts", [])),
                "terms_gt": len(ground_truth.get("terms", []))
            }
            
        return metrics
    
    def to_dataframe(self, extracted: Dict[str, Any]) -> "pandas.DataFrame":
        """
        Преобразовать извлеченные данные в DataFrame.
        
        Args:
            extracted: Извлеченные данные
            
        Returns:
            pandas DataFrame
        """
        import pandas as pd
        
        # Создаем DataFrame для каждой категории
        companies_df = pd.json_normalize(extracted.get("companies", []))
        dates_df = pd.json_normalize(extracted.get("dates", []))
        amounts_df = pd.json_normalize(extracted.get("amounts", []))
        terms_df = pd.json_normalize(extracted.get("terms", []))
        
        return {
            "companies": companies_df,
            "dates": dates_df,
            "amounts": amounts_df,
            "terms": terms_df
        }
