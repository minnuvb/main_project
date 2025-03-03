import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load trained XGBoost model with error handling
try:
    model = joblib.load('xgboost_model.pkl')
except FileNotFoundError:
    st.error("❌ Model file not found! Please train and save the model first.")
    st.stop()

# Load scaler if feature scaling was used
try:
    scaler = joblib.load('scaler.pkl')  # Load the saved scaler
    scaling_applied = True
except FileNotFoundError:
    scaling_applied = False

# Streamlit UI
st.title(" Loan Eligibility Prediction using XGBoost")
st.write("Enter your details below to check loan approval status.")

# Input Fields
person_age = st.number_input("Person Age", min_value=18, max_value=100, value=25)
person_gender = st.selectbox("Person Gender", ["Male", "Female"])
person_education = st.selectbox("Person Education", ["High School", "Associate", "Bachelor", "Master"])
person_income = st.number_input("Person Income (in USD)", min_value=0, value=50000)
person_emp_exp = st.number_input("Person Employment Experience (Years)", min_value=0, max_value=50, value=5)
person_home_ownership = st.selectbox("Person Home Ownership", ["Own", "Rent", "Mortgage"])
loan_amnt = st.number_input("Loan Amount (in USD)", min_value=500, value=10000)
loan_intent = st.selectbox("Loan Intent", ["EDUCATION", "MEDICAL", "PERSONAL", "HOMEIMPROVEMENT", "VENTURE", "DEBTCONSOLIDATION"])
loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=0.0, max_value=50.0, value=10.0)
loan_percent_income = st.number_input("Loan Percent Income", min_value=0.0, max_value=1.0, value=0.2)
cb_person_cred_hist_length = st.number_input("Credit History Length (Years)", min_value=0, max_value=50, value=5)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults on File", ["No", "Yes"])

# Convert categorical inputs to numerical format
def encode_categorical(value, category_list):
    return category_list.index(value)

encoded_inputs = [
    person_age,
    encode_categorical(person_gender, ["Male", "Female"]),
    encode_categorical(person_education, ["High School", "Associate", "Bachelor", "Master"]),
    person_income,
    person_emp_exp,
    encode_categorical(person_home_ownership, ["Own", "Rent", "Mortgage"]),
    loan_amnt,
    encode_categorical(loan_intent, ["EDUCATION", "MEDICAL", "PERSONAL", "HOMEIMPROVEMENT", "VENTURE", "DEBTCONSOLIDATION"]),
    loan_int_rate,
    loan_percent_income,
    cb_person_cred_hist_length,
    credit_score,
    int(previous_loan_defaults_on_file == "Yes")  # Convert "Yes" to 1, "No" to 0
]

# Ensure feature names match training data
feature_columns = [
    "person_age", "person_gender", "person_education", "person_income", "person_emp_exp",
    "person_home_ownership", "loan_amnt", "loan_intent", "loan_int_rate",
    "loan_percent_income", "cb_person_cred_hist_length", "credit_score", "previous_loan_defaults_on_file"
]

encoded_inputs = pd.DataFrame([encoded_inputs], columns=feature_columns)

# Apply scaling if used during training
if scaling_applied:
    encoded_inputs = pd.DataFrame(scaler.transform(encoded_inputs), columns=feature_columns)

# Predict button
if st.button("🔍 Predict"):
    prediction = model.predict(encoded_inputs)[0]
    result = "✅ Approved" if prediction == 1 else "❌ Rejected"
    st.subheader(f"Loan Status: {result}")
