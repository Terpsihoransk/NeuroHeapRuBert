# Предварительно скачиваем модели в .cache\huggingface\hub\
# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("seara/rubert-tiny2-russian-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("seara/rubert-tiny2-russian-sentiment")

# Скачаем все модели заранее


print("Модели успешно загружены!")