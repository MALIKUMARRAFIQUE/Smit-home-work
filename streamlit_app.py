from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from app import clean_features


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "champion_model.pkl"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Champion model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def predict_churn(model, customer):
    features = clean_features(pd.DataFrame([customer]))
    probability = float(model.predict_proba(features)[0, 1])
    return probability


st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="centered")
st.title("Customer Churn Predictor")
st.write("Enter your customer details below to check the likelihood of churn.")

try:
    model = load_model()
except (FileNotFoundError, OSError) as error:
    st.error(str(error))
    st.stop()

with st.form("customer_details"):
    st.subheader("Customer details")
    customer_id = st.text_input("Customer ID", value="New-Customer")

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
        partner = st.selectbox("Has partner", ["Yes", "No"])
        dependents = st.selectbox("Has dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        phone_service = st.selectbox("Phone service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online backup", ["Yes", "No", "No internet service"])

    with col2:
        device_protection = st.selectbox("Device protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        monthly_charges = st.number_input("Monthly charges", min_value=0.0, value=70.0, step=0.01)
        total_charges = st.number_input("Total charges", min_value=0.0, value=840.0, step=0.01)

    submitted = st.form_submit_button("Check churn risk", type="primary", use_container_width=True)

if submitted:
    customer = {
        "customerID": customer_id,
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    try:
        probability = predict_churn(model, customer)
    except (KeyError, ValueError, TypeError) as error:
        st.error(f"Could not calculate churn risk: {error}")
    else:
        if probability >= 0.5:
            st.error(f"High churn risk: {probability:.1%}")
            st.write("This customer is likely to leave the service.")
        else:
            st.success(f"Low churn risk: {probability:.1%}")
            st.write("This customer is likely to remain with the service.")
