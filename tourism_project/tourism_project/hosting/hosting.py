# import necessary libraries
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

# Initialize HfApi with authentication token
api = HfApi(token=os.getenv("HF_TOKEN"))

repo_id = "SandeepGS/tourism-package-prediction"

try:
    api.repo_info(repo_id=repo_id, repo_type="space")
except RepositoryNotFoundError:
    create_repo(repo_id=repo_id, repo_type="space", space_sdk="streamlit", private=False)

# Upload the deployment folder to Hugging Face as a Space
api.upload_folder(
    folder_path="tourism_project/deployment",  # the local folder containing your files
    repo_id="SandeepGS/tourism-package-prediction",  # the target repo
    repo_type="space",  # dataset, model, or space
    path_in_repo="",  # optional: subfolder path inside the repo
)
