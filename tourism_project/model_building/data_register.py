
# ==========================================
# Import required libraries
# ==========================================
import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# ==========================================
# Configuration
# ==========================================
REPO_ID = "SandeepGS/tourism-package-prediction"
REPO_TYPE = "dataset"
DATA_FOLDER = "tourism_project/data"   # ✅ FIX: correct relative path

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN environment variable is not set.")

api = HfApi(token=HF_TOKEN)

# ==========================================
# Validate local dataset folder
# ==========================================
if not os.path.exists(DATA_FOLDER):
    raise FileNotFoundError(
        f"Data folder '{DATA_FOLDER}' not found. Ensure dataset is saved before running data_register.py."
    )

files = os.listdir(DATA_FOLDER)
if not files:
    raise ValueError(f"Data folder '{DATA_FOLDER}' is empty. Nothing to upload.")

print(f"✅ Found data folder with files: {files}")

# ==========================================
# Step 1: Ensure dataset repo exists
# ==========================================
try:
    api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
    print(f"✅ Dataset repo '{REPO_ID}' already exists.")
except RepositoryNotFoundError:
    print(f"⚠️ Dataset repo '{REPO_ID}' not found. Creating new repo...")
    create_repo(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        private=False,
        token=HF_TOKEN
    )
    print(f"✅ Dataset repo '{REPO_ID}' created.")

# ==========================================
# Step 2: Upload dataset folder (SYNC MODE)
# ==========================================

print("🚀 Uploading dataset folder to Hugging Face...")

api.upload_folder(
    folder_path=DATA_FOLDER,
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
    commit_message="Sync dataset folder from CI pipeline",
)

print("✅ Dataset folder synced successfully with Hugging Face repo.")
