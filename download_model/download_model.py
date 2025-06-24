from huggingface_hub import hf_hub_download
import os

os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

try:
    file = hf_hub_download(
        repo_id="seara/rubert-tiny2-russian-sentiment",
        filename="pytorch_model.bin",
        force_download=True,
        resume_download=True
    )
    print(f"Файл успешно скачан: {file}")
except Exception as e:
    print(f"Ошибка: {str(e)}")