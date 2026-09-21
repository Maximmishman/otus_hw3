"""Модуль для работы с датасетом russian-legal-ner."""

from typing import List, Dict, Any, Optional
from datasets import load_dataset
import random


class LegalDataset:
    """Класс для загрузки и работы с датасетом юридических документов."""
    
    def __init__(self, dataset_name: str = "TryDotAtwo/russian-legal-ner"):
        """
        Инициализация датасета.
        
        Args:
            dataset_name: Название датасета на HuggingFace
        """
        self.dataset_name = dataset_name
        self.dataset = None
        self.train_data = []
        self.test_data = []
        
    def load(self, split: str = "train") -> None:
        """
        Загрузка датасета из HuggingFace.
        
        Args:
            split: Часть датасета для загрузки ('train', 'validation', 'test')
        """
        if self.dataset is None:
            print(f"Загрузка датасета {self.dataset_name}...")
            self.dataset = load_dataset(self.dataset_name)
            print(f"Датасет успешно загружен.")
        
        if split == "train":
            self.train_data = self.dataset["train"]
            print(f"Загружено {len(self.train_data)} документов для обучения")
        elif split == "test":
            self.test_data = self.dataset["test"]
            print(f"Загружено {len(self.test_data)} тестовых документов")
        elif split == "validation":
            validation_data = self.dataset["validation"]
            print(f"Загружено {len(validation_data)} валидационных документов")
            
    def get_sample(self, split: str = "train", index: Optional[int] = None) -> Dict[str, Any]:
        """
        Получить образец документа.
        
        Args:
            split: Часть датасета ('train' или 'test')
            index: Индекс документа (случайный если None)
            
        Returns:
            Словарь с текстом документа
        """
        data = self.train_data if split == "train" else self.test_data
        
        if index is None:
            index = random.randint(0, len(data) - 1)
            
        return {
            "text": data[index]["text"],
            "spans": data[index]["spans"]
        }
    
    def get_random_samples(self, split: str = "train", count: int = 5) -> List[Dict[str, Any]]:
        """
        Получить несколько случайных образцов.
        
        Args:
            split: Часть датасета
            count: Количество образцов
            
        Returns:
            Список словарей с документами
        """
        data = self.train_data if split == "train" else self.test_data
        
        indices = random.sample(range(len(data)), min(count, len(data)))
        return [
            {"text": data[i]["text"], "spans": data[i]["spans"]}
            for i in indices
        ]
    
    def get_documents_without_amount(self, count: int = 3) -> List[str]:
        """
        Получить документы без сумм (для тестирования edge cases).
        
        Args:
            count: Количество документов
            
        Returns:
            Список текстов документов
        """
        documents = []
        
        for doc in self.train_data:
            text = doc["text"]
            if not any(keyword.lower() in text.lower() 
                      for keyword in ["руб", "рублей", "тыс", "тысяч", "$", "доллар"]):
                documents.append(text)
                if len(documents) >= count:
                    break
                    
        return documents
    
    def get_short_documents(self, max_length: int = 200, count: int = 5) -> List[str]:
        """
        Получить короткие документы (для тестирования chunking).
        
        Args:
            max_length: Максимальная длина документа
            count: Количество документов
            
        Returns:
            Список текстов коротких документов
        """
        documents = []
        
        for doc in self.train_data:
            text = doc["text"]
            if len(text) <= max_length:
                documents.append(text)
                if len(documents) >= count:
                    break
                    
        return documents
    
    def __len__(self) -> int:
        """Возвращает количество документов в тренировочной выборке."""
        return len(self.train_data) if self.train_data else 0
