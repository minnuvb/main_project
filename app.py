#user interface
#use streamlit
import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# Load trained XGBoost model which is the best model
model = joblib.load('xgboost_model.pkl')

# Streamlit UI is used here
st.title("Loan Eligibility Prediction using XGBoost")
st.write("Enter your details below to check loan approval status")

# Input Fields
person_age = st.number_input("Person Age", min_value=18, max_value=100, value=25)
person_gender = st.selectbox("Person Gender", ["Male", "Female"])
person_education = st.selectbox("Person Education", ["High School", "Bachelor", "Master", "Associate"])
person_income = st.number_input("Person Income (in USD)", min_value=0, value=50000)
person_emp_exp = st.number_input("Person Employment Experience (Years)", min_value=0, max_value=50, value=5)
person_home_ownership = st.selectbox("Person Home Ownership", ["Own", "Rent", "Mortgage"])
loan_amnt = st.number_input("Loan Amount (in USD)", min_value=500, value=10000)
loan_intent = st.selectbox("Loan Intent", ["EDUCATION", "MEDICAL", "PERSONAL", "HOMEIMPROVEMENT", "VENTURE", "DEBTCONSOLIDATION"])
loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=0.0, max_value=50.0, value=10.0)
loan_percent_income = st.number_input("Loan Percent Income", min_value=0.0, max_value=1.0, value=0.2)
cb_person_cred_hist_length = st.number_input("Credit History Length (Years)", min_value=0, max_value=50, value=5)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults on File", [0, 1])

# Convert categorical inputs to numerical format
def encode_categorical(value, category_list):
    return category_list.index(value)

encoded_inputs = [
    person_age,
    encode_categorical(person_gender, ["Male", "Female"]),
    encode_categorical(person_education, ["High School", "Bachelor", "Master", "Associate"]),
    person_income,
    person_emp_exp,
    encode_categorical(person_home_ownership, ["Own", "Rent", "Mortgage"]),
    loan_amnt,
    encode_categorical(loan_intent, ["EDUCATION", "MEDICAL", "PERSONAL", "HOMEIMPROVEMENT", "VENTURE", "DEBTCONSOLIDATION"]),
    loan_int_rate,
    loan_percent_income,
    cb_person_cred_hist_length,
    credit_score,
    previous_loan_defaults_on_file
]

# Predict button
if st.button("Predict"):
    prediction = model.predict(np.array(encoded_inputs).reshape(1, -1))
    result = "Approved" if prediction[0] == 1 else "Rejected"
    st.subheader(f"Loan Status: {result}")
    
    # SHAP Explanation for Rejected Loans
    if prediction[0] == 0:
        st.write("### Explanation for Loan Rejection")
        explainer = shap.Explainer(model)
        shap_values = explainer(np.array(encoded_inputs).reshape(1, -1))
        
        fig, ax = plt.subplots()
        shap.waterfall_plot(shap_values[0])
        st.pyplot(fig)
