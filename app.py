import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load trained XGBoost model with error handling
try:
    model = joblib.load('xgboost_model.pkl')
except FileNotFoundError:
    st.error("❌ Model file not found! Please train and save the model first.")
    st.stop()

# Load scaler if used
try:
    scaler = joblib.load('scaler.pkl')
    scaling_applied = True
except FileNotFoundError:
    scaling_applied = False

# Navigation
st.sidebar.title("🌐 Navigation")
page = st.sidebar.radio("Go to", ["🏦 Loan Eligibility Prediction", "📊 EMI Estimator", "💰 Loan Amount Calculator"])

if page == "🏦 Loan Eligibility Prediction":
    st.title(":bank: Loan Eligibility Prediction using XGBoost")
    st.write("Enter your details below to check loan approval status (All amounts are in ₹ Rupees)")

    person_age = st.number_input("Person Age", min_value=18, max_value=100, value=25)
    person_gender = st.selectbox("Person Gender", ["Male", "Female"])
    person_education = st.selectbox("Person Education", ["High School", "Associate", "Bachelor", "Master"])
    person_income = st.number_input("Annual Income (₹)", min_value=0, value=500000)
    person_emp_exp = st.number_input("Employment Experience (Years)", min_value=0, max_value=50, value=5)
    person_home_ownership = st.selectbox("Home Ownership", ["Own", "Rent", "Mortgage"])
    loan_amnt = st.number_input("Loan Amount (₹)", min_value=5000, value=100000)
    loan_intent = st.selectbox("Loan Intent", ["EDUCATION", "MEDICAL", "PERSONAL", "HOMEIMPROVEMENT", "VENTURE", "DEBTCONSOLIDATION"])
    loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=50.0, value=10.0)
    loan_percent_income = loan_amnt / person_income if person_income > 0 else 0
    cb_person_cred_hist_length = st.number_input("Credit History Length (Years)", min_value=0, max_value=50, value=5)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
    previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults on File", ["No", "Yes"])

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
        int(previous_loan_defaults_on_file == "Yes")
    ]

    feature_columns = [
        "person_age", "person_gender", "person_education", "person_income", "person_emp_exp",
        "person_home_ownership", "loan_amnt", "loan_intent", "loan_int_rate",
        "loan_percent_income", "cb_person_cred_hist_length", "credit_score", "previous_loan_defaults_on_file"
    ]

    encoded_inputs = pd.DataFrame([encoded_inputs], columns=feature_columns)

    if scaling_applied:
        encoded_inputs = pd.DataFrame(scaler.transform(encoded_inputs), columns=feature_columns)

    if st.button("🔍 Predict"):
        prediction = model.predict(encoded_inputs)[0]
        prediction_proba = model.predict_proba(encoded_inputs)[0]

        result = "✅ Approved" if prediction == 1 else "❌ Rejected"
        confidence = prediction_proba[1] if prediction == 1 else prediction_proba[0]

        st.subheader(f"Loan Status: {result}")
        st.write(f"🔹 Confidence Score: **{confidence:.2f}**")

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feature_impact = dict(zip(feature_columns, importances))
            sorted_features = sorted(feature_impact.items(), key=lambda x: x[1], reverse=True)

            if prediction == 0:
                st.subheader("📉 Why was the loan rejected?")
                st.write("The loan rejection was influenced by the following key factors:")

                top_reasons = sorted_features[:3]
                for feature, importance in top_reasons:
                    reason_text = ""
                    if feature == "credit_score":
                        reason_text = "A low credit score increases the risk of default."
                    elif feature == "loan_int_rate":
                        reason_text = "A high interest rate indicates higher risk."
                    elif feature == "person_income":
                        reason_text = "Lower income may indicate repayment difficulty."
                    elif feature == "previous_loan_defaults_on_file":
                        reason_text = "Past loan defaults hurt your credibility."
                    elif feature == "cb_person_cred_hist_length":
                        reason_text = "Shorter credit history reduces confidence."
                    elif feature == "loan_percent_income":
                        reason_text = "High loan-to-income ratio indicates financial strain."
                    else:
                        reason_text = "This feature significantly influenced the decision."
                    st.write(f"🔸 **{feature.replace('_', ' ').title()}** - {reason_text}")

            fig, ax = plt.subplots()
            ax.barh([f[0].replace('_', ' ').title() for f in sorted_features], [f[1] for f in sorted_features], color="skyblue")
            ax.set_xlabel("Feature Importance")
            ax.set_title("📊 Feature Importance (XGBoost)")
            ax.invert_yaxis()
            st.pyplot(fig)

elif page == "📊 EMI Estimator":
    st.title(":bar_chart: EMI Estimator")
    loan_amount = st.number_input("Loan Amount (₹)", min_value=1000, value=100000)
    interest_rate = st.number_input("Annual Interest Rate (%)", min_value=1.0, value=10.0)
    loan_tenure = st.number_input("Loan Tenure (in years)", min_value=1, value=5)

    if st.button("📈 Calculate EMI"):
        monthly_rate = interest_rate / 12 / 100
        months = loan_tenure * 12
        emi = loan_amount * monthly_rate * ((1 + monthly_rate)**months) / ((1 + monthly_rate)**months - 1)
        st.success(f"Estimated EMI: ₹{emi:.2f} per month")

elif page == "💰 Loan Amount Calculator":
    st.title(":moneybag: Loan Amount Calculator")
    annual_income = st.number_input("Annual Income (₹)", min_value=100000, value=500000)
    max_loan_ratio = st.slider("Loan-to-Income Ratio", min_value=0.1, max_value=0.8, value=0.4)

    if st.button("💼 Estimate Eligible Loan Amount"):
        eligible_loan = annual_income * max_loan_ratio
        st.success(f"Estimated Eligible Loan Amount: ₹{eligible_loan:.2f}")
