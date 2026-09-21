"""Модуль для работы с YandexGPT API."""

import os
import json
import re
from typing import Dict, Any, List, Optional, Tuple
import requests
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()


class YandexGPTClient:
    """Клиент для работы с YandexGPT API."""
    
    API_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        folder_id: Optional[str] = None,
        model: str = "yandexgpt-lite"
    ):
        """
        Инициализация клиента YandexGPT.
        
        Args:
            api_key: API-ключ Yandex Cloud (из переменных окружения по умолчанию)
            folder_id: Идентификатор каталога (из переменных окружения по умолчанию)
            model: Модель (yandexgpt-lite или yandexgpt)
        """
        self.api_key = api_key or os.getenv("YC_API_KEY")
        self.folder_id = folder_id or os.getenv("YC_FOLDER_ID")
        self.model = model
        
        if not self.api_key:
            raise ValueError("API-ключ не найден. Установите переменную окружения YC_API_KEY")
        if not self.folder_id:
            raise ValueError("Folder ID не найден. Установите переменную окружения YC_FOLDER_ID")
            
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {self.api_key}"
        }
        
    def _build_request(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """
        Создать структуру запроса к API.
        
        Args:
            messages: Список сообщений (system, user, assistant)
            temperature: Температура генерации
            max_tokens: Максимальное количество токенов
            
        Returns:
            Структура запроса
        """
        return {
            "modelUri": f"gpt://{self.folder_id}/{self.model}",
            "completionOptions": {
                "stream": False,
                "temperature": temperature,
                "maxTokens": str(max_tokens)
            },
            "messages": messages
        }
    
    def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Отправить запрос к API.
        
        Args:
            request: Структура запроса
            
        Returns:
            Ответ от API
        """
        response = requests.post(self.API_URL, headers=self.headers, json=request)
        response.raise_for_status()
        return response.json()
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 2000
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Сгенерировать ответ от модели.
        
        Args:
            system_prompt: Системный промпт
            user_prompt: Пользовательский промпт (текст)
            temperature: Температура (для NER рекомендуется 0.1-0.3)
            max_tokens: Максимальное количество токенов
            
        Returns:
            Кортеж (ответ, метаданные запроса)
        """
        messages = [
            {"role": "system", "text": system_prompt},
            {"role": "user", "text": user_prompt}
        ]
        
        request = self._build_request(messages, temperature, max_tokens)
        api_response = self._send_request(request)
        
        # Извлечение ответа из структуры API
        result = api_response.get("result", {})
        alternatives = result.get("alternatives", [])
        
        if not alternatives:
            raise ValueError("Пустой ответ от API")
            
        text = alternatives[0].get("message", {}).get("text", "")
        usage = result.get("usage", {})
        
        return text, {
            "input_tokens": usage.get("inputTextTokens", "N/A"),
            "output_tokens": usage.get("completionTokens", "N/A"),
            "total_tokens": usage.get("totalTokens", "N/A"),
            "model_version": result.get("modelVersion", "N/A")
        }
    
    def generate_with_examples(
        self,
        system_prompt: str,
        documents: List[str],
        few_shot_examples: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2000
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Сгенерировать ответы для нескольких документов.
        
        Args:
            system_prompt: Системный промпт
            documents: Список текстов документов
            few_shot_examples: Примеры для few-shot обучения (опционально)
            temperature: Температура
            max_tokens: Максимальное количество токенов
            
        Returns:
            Список кортежей (ответ, метаданные)
        """
        results = []
        
        for doc in documents:
            user_text = doc
            if few_shot_examples:
                user_text = f"{few_shot_examples}\n\nтекст для анализа:\n{doc}"
            
            response, meta = self.generate(
                system_prompt,
                user_text,
                temperature,
                max_tokens
            )
            results.append((response, meta))
            
        return results
