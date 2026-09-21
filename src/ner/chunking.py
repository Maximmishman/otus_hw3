"""Модуль для разбиения документов на части (chunking)."""

import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Chunk:
    """Представляет одну часть документа."""
    text: str
    start_index: int
    end_index: int
    chunk_index: int
    is_complete: bool = True


class DocumentChunker:
    """Класс для разбиения длинных документов на части."""
    
    # Оценка токенов: ~3 символа = 1 токен
    TOKENS_TO_CHARS = 3
    MAX_TOKENS = 4000  # Безопасное значение для YandexGPT-lite
    
    def __init__(
        self,
        max_tokens: int = 4000,
        overlap_ratio: float = 0.1,
        chunk_by: str = "sentences"
    ):
        """
        Инициализация чанкера документов.
        
        Args:
            max_tokens: Максимальное количество токенов в чанке
            overlap_ratio: Доля перекрытия между чанками (0.0-1.0)
            chunk_by: Метод разбиения ('sentences', 'paragraphs', 'chars')
        """
        self.max_tokens = max_tokens
        self.max_chars = max_tokens * self.TOKENS_TO_CHARS
        self.overlap_ratio = overlap_ratio
        self.chunk_by = chunk_by
        self.min_chunk_length = int(self.max_chars * 0.5)  # Минимальная длина чанка
        
    def _split_into_sentences(self, text: str) -> List[str]:
        """Разбить текст на предложения."""
        # Разделение по точкам, вопросительным и восклицательным знакам
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Разбить текст на абзацы."""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _split_into_words(self, text: str, max_words: int) -> List[str]:
        """Разбить текст на чанки по словам."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), max_words):
            chunk = ' '.join(words[i:i + max_words])
            chunks.append(chunk)
            
        return chunks
    
    def chunk(self, text: str, min_length: Optional[int] = None) -> List[Chunk]:
        """
        Разбить документ на части.
        
        Args:
            text: Текст документа
            min_length: Минимальная длина текста (если None, используется self.min_chunk_length)
            
        Returns:
            Список объектов Chunk
        """
        if min_length is None:
            min_length = self.min_chunk_length
            
        text = text.strip()
        
        # Если текст помещается в один чанк
        if len(text) <= self.max_chars:
            return [Chunk(text=text, start_index=0, end_index=len(text), chunk_index=0)]
        
        chunks = []
        
        if self.chunk_by == "sentences":
            sentences = self._split_into_sentences(text)
            current_chunk = ""
            chunk_idx = 0
            start_idx = 0
            
            for sent in sentences:
                test_chunk = current_chunk + " " + sent if current_chunk else sent
                if len(test_chunk) > self.max_chars and current_chunk:
                    chunks.append(Chunk(
                        text=current_chunk.strip(),
                        start_index=start_idx,
                        end_index=start_idx + len(current_chunk),
                        chunk_index=chunk_idx,
                        is_complete=True
                    ))
                    chunk_idx += 1
                    # Добавляем overlap
                    overlap = int(len(current_chunk) * self.overlap_ratio)
                    start_idx += len(current_chunk) - overlap
                    current_chunk = text[start_idx:start_idx + overlap]
                else:
                    current_chunk = test_chunk
                    
            if current_chunk and len(current_chunk) >= min_length:
                chunks.append(Chunk(
                    text=current_chunk.strip(),
                    start_index=start_idx,
                    end_index=len(text),
                    chunk_index=chunk_idx,
                    is_complete=True
                ))
                
        elif self.chunk_by == "paragraphs":
            paragraphs = self._split_into_paragraphs(text)
            current_chunk = ""
            start_idx = 0
            chunk_idx = 0
            
            for para in paragraphs:
                test_chunk = current_chunk + "\n\n" + para if current_chunk else para
                if len(test_chunk) > self.max_chars and current_chunk:
                    chunks.append(Chunk(
                        text=current_chunk.strip(),
                        start_index=start_idx,
                        end_index=start_idx + len(current_chunk),
                        chunk_index=chunk_idx,
                        is_complete=True
                    ))
                    chunk_idx += 1
                    overlap = int(len(current_chunk) * self.overlap_ratio)
                    start_idx += len(current_chunk) - overlap
                    remaining = text[start_idx:]
                    # Ищем начало следующего абзаца
                    para_match = re.search(r'\n\s*\n\s*(.+)', remaining)
                    if para_match:
                        overlap_start = start_idx + para_match.start()
                        overlap_text = text[overlap_start:start_idx + overlap]
                        current_chunk = overlap_text
                    else:
                        current_chunk = remaining[:overlap]
                else:
                    current_chunk = test_chunk
                    
            if current_chunk and len(current_chunk) >= min_length:
                chunks.append(Chunk(
                    text=current_chunk.strip(),
                    start_index=start_idx,
                    end_index=len(text),
                    chunk_index=chunk_idx,
                    is_complete=True
                ))
                
        else:  # chars or words
            # Разбиение по фиксированному количеству символов
            for i in range(0, len(text), int(self.max_chars * (1 - self.overlap_ratio))):
                end_i = min(i + int(self.max_chars * (1 - self.overlap_ratio)), len(text))
                chunk_text = text[i:end_i]
                
                # Пытаемся разбить по слову
                if len(chunk_text) < self.max_chars:
                    chunks.append(Chunk(
                        text=chunk_text,
                        start_index=i,
                        end_index=end_i,
                        chunk_index=len(chunks),
                        is_complete=False
                    ))
                else:
                    # Ищем последнее пробельное слово
                    last_space = chunk_text.rfind(' ')
                    if last_space > self.max_chars / 2:
                        chunk_text = chunk_text[:last_space]
                        end_i = i + last_space
                    
                    chunks.append(Chunk(
                        text=chunk_text,
                        start_index=i,
                        end_index=end_i,
                        chunk_index=len(chunks),
                        is_complete=False
                    ))
                    
        return chunks
    
    def merge_chunks(self, chunks: List[Chunk], max_tokens: Optional[int] = None) -> List[Chunk]:
        """
        Объединить чанки, если они короткие.
        
        Args:
            chunks: Список чанков
            max_tokens: Максимальное количество токенов
            
        Returns:
            Ограниченный список чанков
        """
        if max_tokens is None:
            max_tokens = self.max_tokens
            
        if len(chunks) <= 1:
            return chunks
            
        merged = []
        current_chunk = chunks[0]
        
        for next_chunk in chunks[1:]:
            test_text = current_chunk.text + "\n\n" + next_chunk.text
            if len(test_text) <= max_tokens * self.TOKENS_TO_CHARS:
                # Объединяем
                current_chunk = Chunk(
                    text=test_text,
                    start_index=current_chunk.start_index,
                    end_index=next_chunk.end_index,
                    chunk_index=current_chunk.chunk_index,
                    is_complete=current_chunk.is_complete and next_chunk.is_complete
                )
            else:
                # Добавляем текущий и начинаем новый
                merged.append(current_chunk)
                current_chunk = next_chunk
                
        merged.append(current_chunk)
        return merged
