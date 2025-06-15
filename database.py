# Работа с PostgreSQL

import psycopg2
from config import DB_CONFIG


class ReviewDB:
    def __init__(self):
        self.conn = psycopg2.connect(**DB_CONFIG)

    def fetch_unprocessed(self, limit=100):
        with self.conn.cursor() as cursor:
            cursor.execute("""
                           SELECT id, text
                           FROM reviews
                           WHERE processed = FALSE
                               LIMIT %s
                           """, (limit,))
            return cursor.fetchall()

    def update_review(self, review_id, sentiment, topics, details, is_appropriate):
        with self.conn.cursor() as cursor:
            cursor.execute("""
                           UPDATE reviews
                           SET sentiment      = %s,
                               topics         = %s,
                               details        = %s,
                               is_appropriate = %s,
                               processed      = TRUE
                           WHERE id = %s
                           """, (sentiment, topics, details, is_appropriate, review_id))
        self.conn.commit()