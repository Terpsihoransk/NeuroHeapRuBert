# Конфигурация

import os
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла
load_dotenv()

DB_CONFIG = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432')
}

MODEL_SETTINGS = {
    'sentiment_model': 'seara/rubert-tiny2-russian-sentiment',
    'ner_model': 'DeepPavlov/rubert-base-cased-sentence',
    'profanity_list': 'profanity_ru.txt'  # Файл с матами
}