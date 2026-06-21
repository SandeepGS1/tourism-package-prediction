import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN environment variable is not set.")

api = HfApi(token=HF_TOKEN)

repo_id = "SandeepGS/tourism_package-prediction"
repo_type = "space"
deployment_folder = "tourism_project/deployment"

if not os.path.isdir(deployment_folder):
    raise FileNotFoundError(f"Deployment folder not found: {deployment_folder}")

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating Docker Space...")
    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=False,
        space_sdk="docker",
        token=HF_TOKEN
    )
    print(f"Space '{repo_id}' created.")

api.upload_folder(
    folder_path=deployment_folder,
    repo_id=repo_id,
    repo_type=repo_type,
    commit_message="Update Docker Space deployment files"
)

print("Space deployment upload complete.")
