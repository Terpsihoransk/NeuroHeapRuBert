from database import ReviewDB
from nlp_processor import NLPProcessor
from moderator import Moderator
import logging
import time

logging.basicConfig(level=logging.INFO)

import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'  # Отключаем предупреждение о симлинках
os.environ['HF_HUB_VERBOSITY'] = 'error'  # Уменьшаем уровень логгирования


def main():
    db = ReviewDB()
    nlp = NLPProcessor()
    moderator = Moderator("profanity_ru.txt")

    while True:
        try:
            reviews = db.fetch_unprocessed(limit=5)
            if not reviews:
                logging.info("Нет новых отзывов. Ожидание 30 секунд...")
                time.sleep(30)
                continue

            for review_id, text in reviews:
                try:
                    # Анализ тональности
                    sentiment, score = nlp.get_sentiment(text)

                    # Извлечение сущностей
                    entities = nlp.extract_entities(text)

                    # Определение тем
                    topics = nlp.detect_topics(text, entities)

                    # Модерация
                    is_appropriate = not moderator.contains_profanity(text)

                    # Обновление БД
                    db.update_review(
                        review_id=review_id,
                        sentiment=sentiment,
                        topics=str(topics),
                        details=", ".join([f"{e[0]} ({e[1]})" for e in entities]),
                        is_appropriate=is_appropriate
                    )

                    logging.info(f"Обработан отзыв ID: {review_id}")

                except Exception as e:
                    logging.error(f"Ошибка обработки отзыва {review_id}: {str(e)}")

        except KeyboardInterrupt:
            logging.info("Завершение работы...")
            break


if __name__ == "__main__":
    main()