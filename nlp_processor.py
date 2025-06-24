from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import torch
from typing import List, Tuple


class NLPProcessor:
    def __init__(self):
        # Указываем явно использовать локальный кэш
        self.sentiment_pipeline = pipeline(
            "text-classification",
            model="seara/rubert-tiny2-russian-sentiment",
            local_files_only=True  # Важно! Только локальные файлы
        )

        # Для NER модели
        self.ner_tokenizer = AutoTokenizer.from_pretrained(
            "DeepPavlov/rubert-base-cased-sentence",
            local_files_only=True
        )
        self.ner_model = AutoModelForTokenClassification.from_pretrained(
            "DeepPavlov/rubert-base-cased-sentence",
            local_files_only=True
        )

        # Метки для классификации
        self.sentiment_labels = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
        self.entity_labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-PROD", "I-PROD"]

    def get_sentiment(self, text: str) -> Tuple[str, float]:
        """Анализ тональности с обработкой ошибок"""
        try:
            result = self.sentiment_pipeline(text, truncation=True, max_length=512)[0]
            return result['label'], round(result['score'], 3)
        except Exception as e:
            print(f"Ошибка анализа тональности: {str(e)}")
            return "NEUTRAL", 0.0

    # ... (остальные методы без изменений)