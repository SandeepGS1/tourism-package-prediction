


# tourism_project/model_building/train.py

import os
import json
import warnings
import shutil
from pathlib import Path

import pandas as pd
import joblib
import xgboost as xgb
import mlflow

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.impute import SimpleImputer

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================
DATASET_REPO_ID = "SandeepGS/tourism_package-prediction"
MODEL_REPO_ID = "SandeepGS/tourism_package-prediction"
MODEL_REPO_TYPE = "model"

MODEL_FILE_NAME = "tourism_conversion_predict_model.joblib"
MODEL_CARD_FILE = "README.md"

CLASSIFICATION_THRESHOLD = 0.45
RANDOM_STATE = 42

FAST_CI = os.getenv("FAST_CI", "true").lower() == "true"

if FAST_CI:
    N_ITER = 8
    CV_FOLDS = 3
    N_ESTIMATORS_OPTIONS = [80, 120]
else:
    N_ITER = 20
    CV_FOLDS = 5
    N_ESTIMATORS_OPTIONS = [100, 150, 200]

# ============================================================
# CI-safe MLflow setup
# ============================================================
MLRUNS_DIR = Path("mlruns").resolve()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file://{MLRUNS_DIR}")
EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "tourism-package-prediction-experiment-ci"
)

# Defensive cleanup for broken leftover folder
bad_artifacts_dir = MLRUNS_DIR / "artifacts"
if bad_artifacts_dir.exists() and bad_artifacts_dir.is_dir():
    shutil.rmtree(bad_artifacts_dir, ignore_errors=True)

MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
    mlflow.create_experiment(EXPERIMENT_NAME)

mlflow.set_experiment(EXPERIMENT_NAME)

# ============================================================
# Hugging Face authentication
# ============================================================
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN environment variable is not set.")

api = HfApi(token=HF_TOKEN)

# ============================================================
# Load prepared datasets
# ============================================================
Xtrain_path = f"hf://datasets/{DATASET_REPO_ID}/Xtrain.csv"
Xtest_path = f"hf://datasets/{DATASET_REPO_ID}/Xtest.csv"
ytrain_path = f"hf://datasets/{DATASET_REPO_ID}/ytrain.csv"
ytest_path = f"hf://datasets/{DATASET_REPO_ID}/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze("columns")
ytest = pd.read_csv(ytest_path).squeeze("columns")

print(f"Training set loaded: X={Xtrain.shape}, y={ytrain.shape}")
print(f"Test set loaded: X={Xtest.shape}, y={ytest.shape}")

# ============================================================
# Feature groups
# ============================================================
nominal_features = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
]
ordinal_features = ["Designation"]
numeric_features = [
    "Age",
    "CityTier",
    "DurationOfPitch",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome",
]

# ============================================================
# Preprocessing
# ============================================================
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

nominal_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ]
)

ordinal_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OrdinalEncoder(
                categories=[["Executive", "Manager", "Senior Manager", "AVP", "VP"]],
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("nom", nominal_transformer, nominal_features),
        ("ord", ordinal_transformer, ordinal_features),
    ]
)

# ============================================================
# Class imbalance handling
# ============================================================
class_counts = ytrain.value_counts()
if 0 not in class_counts.index or 1 not in class_counts.index:
    raise ValueError("Target variable must contain both classes 0 and 1.")

class_weight = class_counts[0] / class_counts[1]

# ============================================================
# Model
# ============================================================
xgb_model = xgb.XGBClassifier(
    eval_metric="logloss",
    scale_pos_weight=class_weight,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    tree_method="hist",
)

model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("xgbclassifier", xgb_model),
    ]
)

param_distributions = {
    "xgbclassifier__n_estimators": N_ESTIMATORS_OPTIONS,
    "xgbclassifier__max_depth": [3, 4],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__subsample": [0.8, 1.0],
    "xgbclassifier__colsample_bytree": [0.8, 1.0],
}

search = RandomizedSearchCV(
    estimator=model_pipeline,
    param_distributions=param_distributions,
    n_iter=N_ITER,
    cv=CV_FOLDS,
    scoring="f1",
    n_jobs=-1,
    verbose=1,
    random_state=RANDOM_STATE,
    refit=True,
)

# ============================================================
# Train + log
# ============================================================
with mlflow.start_run(run_name="xgboost_fast_ci_search"):
    mlflow.log_param("fast_ci", FAST_CI)
    mlflow.log_param("n_iter", N_ITER)
    mlflow.log_param("cv_folds", CV_FOLDS)
    mlflow.log_param("tracking_uri", MLFLOW_TRACKING_URI)
    mlflow.log_param("dataset_repo_id", DATASET_REPO_ID)
    mlflow.log_param("model_repo_id", MODEL_REPO_ID)

    search.fit(Xtrain, ytrain)

    best_model = search.best_estimator_
    mlflow.log_params(search.best_params_)
    mlflow.log_metric("best_cv_f1", float(search.best_score_))

    cv_results = pd.DataFrame(search.cv_results_)
    cv_results.to_csv("cv_results.csv", index=False)
    mlflow.log_artifact("cv_results.csv")

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]

    y_pred_train = (y_pred_train_proba >= CLASSIFICATION_THRESHOLD).astype(int)
    y_pred_test = (y_pred_test_proba >= CLASSIFICATION_THRESHOLD).astype(int)

    metrics = {
        "train_accuracy": accuracy_score(ytrain, y_pred_train),
        "train_precision": precision_score(ytrain, y_pred_train, zero_division=0),
        "train_recall": recall_score(ytrain, y_pred_train, zero_division=0),
        "train_f1_score": f1_score(ytrain, y_pred_train, zero_division=0),
        "test_accuracy": accuracy_score(ytest, y_pred_test),
        "test_precision": precision_score(ytest, y_pred_test, zero_division=0),
        "test_recall": recall_score(ytest, y_pred_test, zero_division=0),
        "test_f1_score": f1_score(ytest, y_pred_test, zero_division=0),
    }
    mlflow.log_metrics(metrics)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    with open("train_classification_report.json", "w") as f:
        json.dump(train_report, f, indent=2)

    with open("test_classification_report.json", "w") as f:
        json.dump(test_report, f, indent=2)

    mlflow.log_artifact("train_classification_report.json")
    mlflow.log_artifact("test_classification_report.json")

    joblib.dump(best_model, MODEL_FILE_NAME)
    mlflow.log_artifact(MODEL_FILE_NAME, artifact_path="model")

# ============================================================
# Ensure HF model repo exists
# ============================================================
try:
    api.repo_info(repo_id=MODEL_REPO_ID, repo_type=MODEL_REPO_TYPE)
    print(f"Model repo '{MODEL_REPO_ID}' already exists.")
except RepositoryNotFoundError:
    create_repo(
        repo_id=MODEL_REPO_ID,
        repo_type=MODEL_REPO_TYPE,
        private=False,
        token=HF_TOKEN,
    )
    print(f"Model repo '{MODEL_REPO_ID}' created.")

# ============================================================
# Upload model artifact
# ============================================================
api.upload_file(
    path_or_fileobj=MODEL_FILE_NAME,
    path_in_repo=MODEL_FILE_NAME,
    repo_id=MODEL_REPO_ID,
    repo_type=MODEL_REPO_TYPE,
)

# ============================================================
# Create and upload model card
# ============================================================
with open(MODEL_CARD_FILE, "w", encoding="utf-8") as f:
    f.write(f"""---
tags:
- tourism
- xgboost
- classification
- ci-fast
library_name: scikit-learn
pipeline_tag: tabular-classification
---

# Tourism Package Prediction Model

Fast-CI optimized training run.

## Best Params
{json.dumps(search.best_params_, indent=2)}

## Metrics
{json.dumps(metrics, indent=2)}
""")

api.upload_file(
    path_or_fileobj=MODEL_CARD_FILE,
    path_in_repo=MODEL_CARD_FILE,
    repo_id=MODEL_REPO_ID,
    repo_type=MODEL_REPO_TYPE,
)

print("Fast training run completed successfully.")
