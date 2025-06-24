import requests
try:
    response = requests.get("https://huggingface.co", timeout=5)
    print(f"Подключение к Hugging Face Hub работает! Статус: {response.status_code}")
except Exception as e:
    print(f"Ошибка подключения: {str(e)}")