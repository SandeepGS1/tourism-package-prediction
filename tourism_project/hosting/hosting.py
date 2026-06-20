# import necessary libraries
from huggingface_hub import HfApi
import os

# Initialize HfApi with authentication token
api = HfApi(token=os.getenv("HF_TOKEN"))

# Upload the deployment folder to Hugging Face as a Space
api.upload_folder(
    folder_path="deployment",  # the local folder containing your files
    repo_id="SandeepGS/tourism-package-prediction",  # the target repo
    repo_type="space",  # dataset, model, or space
    path_in_repo="",  # optional: subfolder path inside the repo
)
