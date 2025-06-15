# Основной скрипт

from database import ReviewDB
from nlp_processor import NLPProcessor
from moderator import Moderator
from config import MODEL_SETTINGS
import time
import logging

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('review_processor.log'),
        logging.StreamHandler()
    ]
)


def process_batch(db, nlp, moderator, batch_size=10):
    """Обработка пачки отзывов"""
    reviews = db.fetch_unprocessed(limit=batch_size)
    if not reviews:
        logging.info("Нет новых отзывов для обработки")
        return False

    for review_id, review_text in reviews:
        try:
            logging.info(f"Начата обработка отзыва ID: {review_id}")

            # Шаг 1: Анализ тональности
            sentiment, confidence = nlp.get_sentiment(review_text)

            # Шаг 2: Извлечение сущностей
            entities = nlp.extract_entities(review_text)

            # Шаг 3: Определение тем
            topics = nlp.detect_topics(review_text, entities)

            # Шаг 4: Модерация
            is_appropriate = not moderator.contains_profanity(review_text)

            # Шаг 5: Обновление в БД
            db.update_review(
                review_id=review_id,
                sentiment=sentiment,
                topics=str(topics),
                details=", ".join([f"{e[0]} ({e[1]})" for e in entities]),
                is_appropriate=is_appropriate
            )

            logging.info(f"Успешно обработан отзыв ID: {review_id}")
            logging.debug(f"Результаты: {sentiment=}, {topics=}, {is_appropriate=}")

        except Exception as e:
            logging.error(f"Ошибка обработки отзыва ID: {review_id}: {str(e)}")
            continue

    return True


def main():
    try:
        logging.info("Инициализация сервиса обработки отзывов...")

        # Инициализация компонентов
        db = ReviewDB()
        nlp = NLPProcessor()
        moderator = Moderator(MODEL_SETTINGS['profanity_list'])

        logging.info("Сервис успешно инициализирован, начало обработки...")

        # Основной цикл обработки
        while True:
            try:
                if not process_batch(db, nlp, moderator):
                    time.sleep(60)  # Пауза при отсутствии новых отзывов
            except KeyboardInterrupt:
                logging.info("Получен сигнал прерывания, завершение работы...")
                break
            except Exception as e:
                logging.error(f"Критическая ошибка в основном цикле: {str(e)}")
                time.sleep(300)  # Пауза при критических ошибках

    except Exception as e:
        logging.critical(f"Фатальная ошибка при запуске: {str(e)}")
    finally:
        if 'db' in locals():
            db.conn.close()
        logging.info("Сервис остановлен")


if __name__ == "__main__":
    main()