# import necessary libraries
import pandas as pd
import os
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import joblib

import mlflow

# Set up MLFlow tracking
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")\nmlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("tourism-package-prediction-experiment")

# Initialize Hugging Face API
api = HfApi(token=os.getenv("HF_TOKEN"))

# Load the dataset from Hugging Face dataset repo
Xtrain_path = "hf://datasets/SandeepGS/tourism_package-prediction/Xtrain.csv"
Xtest_path = "hf://datasets/SandeepGS/tourism_package-prediction/Xtest.csv"
ytrain_path = "hf://datasets/SandeepGS/tourism_package-prediction/ytrain.csv"
ytest_path = "hf://datasets/SandeepGS/tourism_package-prediction/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

print(f"Training set loaded: X={Xtrain.shape}, y={ytrain.shape}")
print(f"Test set loaded: X={Xtest.shape}, y={ytest.shape}")

# Define feature groups
nominal_features = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus']
ordinal_features = ['Designation']
numeric_features = [
    'Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting',
    'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips',
    'Passport', 'PitchSatisfactionScore', 'OwnCar',
    'NumberOfChildrenVisiting', 'MonthlyIncome'
]

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),  # Keep numeric as-is
        ('nom', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), nominal_features),
        ('ord', OrdinalEncoder(
            categories=[['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP']],
            handle_unknown='use_encoded_value', unknown_value=-1
        ), ordinal_features)
    ]
)

# Set the clas weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(
    eval_metric='logloss', scale_pos_weight=class_weight, random_state=42
)

# Define hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [150, 175, 200, 225, 250],  # number of tree to build
    'xgbclassifier__max_depth': [3, 4, 5],  # maximum depth of each tree
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
    # percentage of attributes to be considered (randomly) for each tree
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
    # percentage of attributes to be considered (randomly) for each level of a tree
    'xgbclassifier__learning_rate': [0.05, 0.1, 0.15],  # learning rate
    'xgbclassifier__reg_lambda': [0.5, 0.6, 0.7],  # L2 regularization factor
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Start MLFlow run
with mlflow.start_run():
    # Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    # define the classification threshold
    classification_threshold = 0.45

    # predict the training data using the best model
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    # predict the test data using the best model
    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log the metrics for the best model
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    # Save the model locally
    model_path = "tourism_conversion_predict_model.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Create model card
    model_card_content = f"""---
tags:
- tourism
- xgboost
- classification
- customer-prediction
library_name: scikit-learn
pipeline_tag: tabular-classification
---

# Tourism Package Prediction Model

## Model Description

This model predicts whether a customer will purchase a Wellness Tourism Package from "Visit with Us" travel company. It uses XGBoost classifier with a custom preprocessing pipeline to handle both numeric and categorical features.

## Intended Use

**Primary Use:** Identify potential customers for the Wellness Tourism Package to optimize marketing outreach and improve conversion rates.

**Users:** Sales and marketing teams at travel companies.

**Out-of-scope:** This model should not be used for discriminatory purposes or decisions that could significantly impact individuals' lives beyond marketing preferences.

## Training Data

- **Dataset:** Tourism package purchase history
- **Features:** 18 features including customer demographics, travel preferences, and sales interaction data
  - 12 numeric features (Age, CityTier, MonthlyIncome, etc.)
  - 6 categorical features (Gender, Occupation, Designation, etc.)
- **Target:** Binary classification (ProdTaken: 0 = No purchase, 1 = Purchase)
- **Training Set:** {Xtrain.shape[0]} samples
- **Test Set:** {Xtest.shape[0]} samples
- **Class Imbalance:** Handled using `scale_pos_weight` parameter

## Model Architecture

**Algorithm:** XGBoost Classifier with sklearn preprocessing pipeline

**Preprocessing:**
- Numeric features: Passthrough (no transformation)
- Nominal categorical features: OrdinalEncoder
- Ordinal feature (Designation): OrdinalEncoder with hierarchy (Executive → Manager → Senior Manager → AVP → VP)

**Best Hyperparameters:**
{chr(10).join([f"- {k.replace('xgbclassifier__', '')}: {v}" for k, v in grid_search.best_params_.items()])}

**Classification Threshold:** 0.45 (optimized for F1-score)

## Performance Metrics

### Training Set
- **Accuracy:** {train_report['accuracy']:.4f}
- **Precision:** {train_report['1']['precision']:.4f}
- **Recall:** {train_report['1']['recall']:.4f}
- **F1-Score:** {train_report['1']['f1-score']:.4f}

### Test Set
- **Accuracy:** {test_report['accuracy']:.4f}
- **Precision:** {test_report['1']['precision']:.4f}
- **Recall:** {test_report['1']['recall']:.4f}
- **F1-Score:** {test_report['1']['f1-score']:.4f}

## How to Use

```python
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

# Download the model
model_path = hf_hub_download(
    repo_id="SandeepGS/tourism_package-prediction",
    filename="tourism_conversion_predict_model.joblib",
    repo_type="model"
)

# Load the model
model = joblib.load(model_path)

# Prepare input data (must match training feature order)
input_data = pd.DataFrame([{{
    'Age': 35,
    'CityTier': 1,
    'DurationOfPitch': 15,
    'NumberOfPersonVisiting': 3,
    'NumberOfFollowups': 3,
    'PreferredPropertyStar': 4.0,
    'NumberOfTrips': 3,
    'Passport': 1,
    'PitchSatisfactionScore': 3,
    'OwnCar': 1,
    'NumberOfChildrenVisiting': 1,
    'MonthlyIncome': 22000,
    'TypeofContact': 'Self Enquiry',
    'Occupation': 'Salaried',
    'Gender': 'Male',
    'ProductPitched': 'Basic',
    'MaritalStatus': 'Married',
    'Designation': 'Manager'
}}])

# Get prediction probability
prediction_proba = model.predict_proba(input_data)[0, 1]

# Apply custom threshold
prediction = (prediction_proba >= 0.45).astype(int)

print(f"Purchase Probability: {{prediction_proba:.2%}}")
print(f"Prediction: {{'Will Purchase' if prediction == 1 else 'Will Not Purchase'}}")
```

## Training Procedure

1. **Data Preparation:** 80/20 train-test split with stratification
2. **Hyperparameter Tuning:** GridSearchCV with 5-fold cross-validation
3. **Optimization Metric:** F1-Score (to balance precision and recall)
4. **Experiment Tracking:** MLflow for logging parameters and metrics

## Limitations and Considerations

- The model is trained on historical data and may not generalize to significantly different customer populations
- Performance depends on data quality and feature completeness
- Class imbalance handled but may still affect predictions on minority class
- Custom threshold of 0.45 optimized for current dataset; may need adjustment for different use cases
- Model assumes input features are in the exact order and format as training data

## Ethical Considerations

- Ensure model is used responsibly for marketing purposes only
- Regularly monitor for bias in predictions across different demographic groups
- Respect customer privacy and comply with data protection regulations
- Provide opt-out mechanisms for customers who don't wish to be contacted

## Model Card Authors

Sriram Narasimhan

## Model Card Contact

For questions or issues, please open an issue in the model repository.
"""

    # Save model card
    with open("README.md", "w") as f:
        f.write(model_card_content)
    print("Model card created: README.md")

    # Upload to Hugging Face
    repo_id = "SandeepGS/tourism_package-prediction"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # Upload model file
    api.upload_file(
        path_or_fileobj="tourism_conversion_predict_model.joblib",
        path_in_repo="tourism_conversion_predict_model.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print("Model file uploaded to Hugging Face")

    # Upload model card
    api.upload_file(
        path_or_fileobj="README.md",
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type=repo_type,
    )
