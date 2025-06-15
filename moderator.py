# Автомодерация

import re


class Moderator:
    def __init__(self, profanity_file):
        with open(profanity_file, 'r', encoding='utf-8') as f:
            self.profanity_list = [word.strip().lower() for word in f.readlines()]

    def contains_profanity(self, text):
        text = text.lower()
        # Проверка точных совпадений
        for word in self.profanity_list:
            if re.search(rf'\b{re.escape(word)}\b', text):
                return True
        return False