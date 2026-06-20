# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/SandeepGS/tourism-package-prediction/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop the CustomerID column as it is an identifier
df.drop(columns=['CustomerID'], inplace=True)

# Fix the Gender column and change Fe Male to Female
df['Gender'] = df['Gender'].replace('Fe Male', 'Female')

# define the categorical, numeric features and target variable
target_variable = "ProdTaken"

# Define feature set and target variable
X = df.drop('ProdTaken', axis=1)
y = df[target_variable]

# Split dataset into train and test, stratified on target variable as it is imbalanced
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save the prepared data to CSV files
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print(f"Training set: X={Xtrain.shape}, y={ytrain.shape}")
print(f"Test set: X={Xtest.shape}, y={ytest.shape}")

# List of created files
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

# Upload the prepared data files to Hugging Face dataset repo
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id="nsriram78/tourism-package-prediction",
        repo_type="dataset",
    )

print("Data preparation complete! All 4 files uploaded.")
