
# tourism_project/hosting/hosting.py

import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN environment variable is not set.")

api = HfApi(token=HF_TOKEN)

# Keep this consistent with the repo ID you fixed elsewhere
repo_id = "SandeepGS/tourism_package-prediction"
repo_type = "space"

# Correct path relative to repo root in GitHub Actions
deployment_folder = "tourism_project/deployment"

if not os.path.isdir(deployment_folder):
    raise FileNotFoundError(
        f"Deployment folder not found: {deployment_folder}. "
        "Expected repo structure: tourism_project/deployment"
    )

# Ensure the Space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists.")
except RepositoryNotFoundError:
    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=False,
        space_sdk="streamlit",
        token=HF_TOKEN
    )
    print(f"Space '{repo_id}' created.")

# Upload deployment files to Space
api.upload_folder(
    folder_path=deployment_folder,
    repo_id=repo_id,
    repo_type=repo_type,
    commit_message="Update Streamlit Space deployment files"
)

print("Space deployment upload complete.")
