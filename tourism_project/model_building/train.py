
import pandas as pd
import joblib
from huggingface_hub import HfApi
import os

df = pd.read_csv("hf://datasets/SandeepGS/tourism_package-prediction/tourism.csv")

X = df.drop("ProdTaken", axis=1)
y = df["ProdTaken"]

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X, y)

joblib.dump(model, "model.joblib")

HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)

api.upload_file(
    path_or_fileobj="model.joblib",
    path_in_repo="model.joblib",
    repo_id="SandeepGS/tourism_package-prediction",
    repo_type="model"
)

print("Model trained & uploaded")
