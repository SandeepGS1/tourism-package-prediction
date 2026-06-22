
# ==========================================
# Import required libraries
# ==========================================
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError


# ==========================================
# Configuration
# ==========================================
DATASET_REPO_ID = "SandeepGS/tourism_package-prediction"
DATA_FILE_NAME = "tourism.csv"

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN environment variable is not set.")

api = HfApi(token=HF_TOKEN)

print(f"Downloading dataset '{DATA_FILE_NAME}' from Hugging Face dataset repo...")
try:
    local_csv_path = hf_hub_download(
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        filename=DATA_FILE_NAME,
        token=HF_TOKEN
    )
    df = pd.read_csv(local_csv_path)
except Exception as e:
    raise RuntimeError(
        f"Failed to load dataset '{DATA_FILE_NAME}' from dataset repo '{DATASET_REPO_ID}'."
    ) from e

print("✅ Dataset loaded successfully.")
print(f"Initial shape: {df.shape}")


# ==========================================
# Data Cleaning
# ==========================================

# Remove unnamed columns
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# Drop CustomerID if exists
if "CustomerID" in df.columns:
    df.drop(columns=["CustomerID"], inplace=True)

# Fix data inconsistencies
if "Gender" in df.columns:
    df["Gender"] = df["Gender"].replace("Fe Male", "Female")

# Basic validation
if "ProdTaken" not in df.columns:
    raise ValueError("Target column 'ProdTaken' not found in dataset.")

print("✅ Data cleaning complete.")
print(f"Post-cleaning shape: {df.shape}")


# ==========================================
# Train-Test Split
# ==========================================
X = df.drop("ProdTaken", axis=1)
y = df["ProdTaken"]

# Validate target
if y.nunique() < 2:
    raise ValueError("Target variable must have at least 2 classes.")

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("✅ Train-test split complete.")
print(f"Training set: X={Xtrain.shape}, y={ytrain.shape}")
print(f"Test set: X={Xtest.shape}, y={ytest.shape}")


# ==========================================
# Save files locally
# ==========================================
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)

# Ensure 1D series saved properly
ytrain.to_frame(name="ProdTaken").to_csv("ytrain.csv", index=False)
ytest.to_frame(name="ProdTaken").to_csv("ytest.csv", index=False)

print("✅ CSV files saved locally.")


# ==========================================
# Upload to Hugging Face dataset repo
# ==========================================
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

print("Uploading files to Hugging Face dataset repo...")

# Ensure dataset repo exists
try:
    api.repo_info(repo_id=DATASET_REPO_ID, repo_type="dataset")
    print(f"Dataset repo '{DATASET_REPO_ID}' found.")
except RepositoryNotFoundError:
    raise RuntimeError(
        f"Dataset repo '{DATASET_REPO_ID}' does not exist. Run data_register.py first."
    )

# Upload files
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id=DATASET_REPO_ID,
        repo_type="dataset"
    )
    print(f"✅ Uploaded: {file_path}")

print("✅ Data preparation complete. All files uploaded successfully.")
