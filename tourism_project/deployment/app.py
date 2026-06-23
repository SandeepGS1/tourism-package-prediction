
# tourism_project/deployment/app.py

import streamlit as st
import pandas as pd
import joblib
import traceback
from huggingface_hub import hf_hub_download


import os

print("✅ App starting...")
print("Working dir:", os.getcwd())
print("Files:", os.listdir())

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="🌍",
    layout="centered"
)

st.title("🌍 Tourism Package Prediction")
st.markdown("Predict whether a customer will purchase a tourism package.")

# ============================================================
# Load Model Safely
# ============================================================
@st.cache_resource
def load_model():
    try:
        print("🔄 Downloading model...")

        model_path = hf_hub_download(
            repo_id="SandeepGS/tourism_package-prediction",
            filename="tourism_conversion_predict_model.joblib",
            repo_type="model"
        )

        print("✅ Model downloaded:", model_path)

        model = joblib.load(model_path)

        print("✅ Model loaded successfully")
        return model

    except Exception as e:
        print("❌ MODEL LOAD ERROR:", str(e))
        raise e

try:
    model = load_model()
    print("✅ App fully initialized")

except Exception as e:
    st.error("❌ Model failed to load")
    st.text(str(e))
    print("❌ MODEL LOAD FAILED:", str(e))

    st.warning("App is running without model. Please check logs.")
    
    # ✅ DO NOT block — just safely exit render cycle
    st.stop()



# ============================================================
# Helper Function → Input DataFrame
# ============================================================
def prepare_input(data):
    return pd.DataFrame([data])


# ============================================================
# User Inputs
# ============================================================
st.header("🧾 Enter Customer Details")

age = st.number_input("Age", 18, 100, 30)
city_tier = st.selectbox("City Tier", [1, 2, 3])

occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Large Business", "Free Lancer"]
)

gender = st.selectbox("Gender", ["Male", "Female"])

product_pitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
)

marital_status = st.selectbox(
    "Marital Status",
    ["Married", "Unmarried"]
)

designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)

monthly_income = st.number_input("Monthly Income", 1000, 500000, 30000)

num_trips = st.number_input("Number of Trips", 0, 20, 3)

passport = st.selectbox("Has Passport", [0, 1])
own_car = st.selectbox("Owns Car", [0, 1])

num_children = st.number_input("Number of Children Visiting", 0, 10, 0)

followups = st.number_input("Number of Followups", 0, 10, 2)

pitch_duration = st.number_input("Duration of Pitch", 0, 200, 20)

preferred_star = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])

pitch_score = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])

num_person_visiting = st.number_input("Number of Persons Visiting", 1, 10, 2)

type_of_contact = st.selectbox(
    "Type of Contact",
    ["Self Enquiry", "Company Invited"]
)

# ============================================================
# Prediction
# ============================================================
if st.button("🚀 Predict"):

    try:
        input_data = {
            "Age": age,
            "CityTier": city_tier,
            "Occupation": occupation,
            "Gender": gender,
            "ProductPitched": product_pitched,
            "MaritalStatus": marital_status,
            "Designation": designation,
            "MonthlyIncome": monthly_income,
            "NumberOfTrips": num_trips,
            "Passport": passport,
            "OwnCar": own_car,
            "NumberOfChildrenVisiting": num_children,
            "NumberOfFollowups": followups,
            "DurationOfPitch": pitch_duration,
            "PreferredPropertyStar": preferred_star,
            "PitchSatisfactionScore": pitch_score,
            "NumberOfPersonVisiting": num_person_visiting,
            "TypeofContact": type_of_contact,
        }

        df_input = prepare_input(input_data)

        prediction = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0][1]

        st.subheader("📊 Prediction Result")

        if prediction == 1:
            st.success(f"✅ Customer is likely to PURCHASE (Confidence: {probability:.2f})")
        else:
            st.warning(f"❌ Customer is unlikely to purchase (Confidence: {probability:.2f})")

    except Exception:
        st.error("❌ Prediction failed")
        st.text(traceback.format_exc())

st.write("✅ App running... waiting for input")
