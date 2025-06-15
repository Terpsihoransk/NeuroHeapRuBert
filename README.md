## Для старта введи в терминал: 

`` pip install torch transformers psycopg2-binary scikit-learn pandas numpy ``

## Добавляем работу с файлом .env:

`` pip install python-dotenv ``

## Тестовая таблица:

``
CREATE TABLE reviews (
    id SERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    sentiment VARCHAR(20),
    topics TEXT[],
    details TEXT[],
    is_appropriate BOOLEAN,
    processed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
``

## Индексы:

``
CREATE INDEX idx_reviews_processed ON reviews(processed);
CREATE INDEX idx_reviews_created_at ON reviews(created_at);
``

## Тестовые отзывы:

``
INSERT INTO reviews (text, processed) VALUES
('Кредитная карта с отличными условиями, но мобильное приложение иногда глючит', FALSE),
('Одобрили ипотеку на день позже обещанного срока, но условия соответствуют договору', FALSE),
('Ужасный сервис! С меня трижды списали комиссию за перевод без предупреждения', FALSE),
('Стандартный дебетовый счет, ничего особенного - как у всех банков', FALSE),
('Очень доволен инвестиционным продуктом, всё прозрачно и доходность выше рынка', FALSE);
``

## Для работы в оффлайн нужно:
1. скачать модели:
`` from transformers import pipeline
pipeline('text-classification', model='seara/rubert-tiny2-russian-sentiment') ``
2. запускать скрипт с параметром:
`` HF_DATASETS_OFFLINE=1 ``