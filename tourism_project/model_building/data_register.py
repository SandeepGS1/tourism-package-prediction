
import os
from huggingface_hub import HfApi

HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)

api.upload_folder(
    folder_path="tourism_project/data",
    repo_id="SandeepGS/tourism_package-prediction",
    repo_type="dataset"
)
print("Dataset uploaded")
