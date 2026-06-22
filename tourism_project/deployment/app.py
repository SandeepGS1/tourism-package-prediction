
import streamlit as st
from huggingface_hub import hf_hub_download
import joblib

model_path = hf_hub_download(
    repo_id="SandeepGS/tourism_package-prediction",
    filename="model.joblib"
)

model = joblib.load(model_path)

st.title("Prediction App")

st.write("Model loaded successfully!")
