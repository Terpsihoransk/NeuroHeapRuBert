# NLP обработка

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
import torch
from typing import Tuple, List


class NLPProcessor:
    def __init__(self, cache_dir="models_cache"):
        # Инициализация модели для тональности
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(
            "seara/rubert-tiny2-russian-sentiment",
            cache_dir=cache_dir
        )
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
            "seara/rubert-tiny2-russian-sentiment",
            cache_dir=cache_dir
        )

        # Инициализация модели для NER (извлечение сущностей)
        self.ner_tokenizer = AutoTokenizer.from_pretrained(
            "DeepPavlov/rubert-base-cased-sentence",
            cache_dir=cache_dir
        )
        self.ner_model = AutoModelForTokenClassification.from_pretrained(
            "DeepPavlov/rubert-base-cased-sentence",
            cache_dir=cache_dir
        )

        # Метки для классификации тональности
        self.sentiment_labels = ["NEGATIVE", "NEUTRAL", "POSITIVE"]

        # Метки для NER
        self.entity_labels = [
            "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC",
            "B-PROD", "I-PROD", "B-DATE", "I-DATE"
        ]

    def get_sentiment(self, text: str) -> Tuple[str, float]:
        """Определение тональности текста"""
        inputs = self.sentiment_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.sentiment_model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_label = torch.argmax(probs).item()

        return self.sentiment_labels[pred_label], round(probs[0][pred_label].item(), 3)

    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """Извлечение именованных сущностей"""
        inputs = self.ner_tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.ner_model(**inputs)

        predictions = torch.argmax(outputs.logits, dim=2)[0].numpy()
        tokens = self.ner_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        entities = []
        current_entity = ""
        current_label = ""

        for token, prediction in zip(tokens, predictions):
            label = self.entity_labels[prediction]

            if token.startswith("##"):
                current_entity += token[2:]
            elif label.startswith("B-"):
                if current_entity:
                    entities.append((current_entity, current_label[2:]))
                current_entity = token
                current_label = label
            elif label.startswith("I-") and current_label[2:] == label[2:]:
                current_entity += " " + token
            else:
                if current_entity:
                    entities.append((current_entity, current_label[2:]))
                current_entity = ""
                current_label = ""

        if current_entity:
            entities.append((current_entity, current_label[2:]))

        return entities

    def detect_topics(self, text: str, entities: list) -> List[str]:
        """Определение тематик отзыва"""
        keywords = {
            'карта': ['карта', 'пластик', 'платеж', 'кредитка'],
            'приложение': ['приложение', 'мобильное', 'ios', 'android', 'app'],
            'сервис': ['обслуживание', 'поддержка', 'консультация', 'менеджер'],
            'платежи': ['платеж', 'оплата', 'перевод', 'списание', 'комиссия'],
            'кэшбэк': ['кэшбек', 'cashback', 'бонус', 'возврат']
        }

        detected_topics = set()
        text_lower = text.lower()

        for topic, words in keywords.items():
            if any(word in text_lower for word in words):
                detected_topics.add(topic)

        # Добавляем темы из извлеченных сущностей
        for entity, label in entities:
            if label == "PROD":
                detected_topics.add(entity.lower())

        return sorted(detected_topics)