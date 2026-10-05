import os
import sys
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.predict import load_artifacts, predict_loan_risk

# Page Configuration
st.set_page_config(
    page_title="Loan Default Risk Dashboard",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Risk & Loan Default Prediction System")
st.markdown("Evaluate applicant risk profile using our trained **XGBoost** model.")

# Sidebar - Applicant Input Form
st.sidebar.header("Applicant Information")

person_age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=28)
person_income = st.sidebar.number_input("Annual Income ($)", min_value=1000, max_value=1000000, value=60000, step=1000)
person_home_ownership = st.sidebar.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
person_emp_length = st.sidebar.number_input("Employment Length (Years)", min_value=0.0, max_value=50.0, value=4.0, step=0.5)

loan_intent = st.sidebar.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
loan_grade = st.sidebar.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
loan_amnt = st.sidebar.number_input("Loan Amount ($)", min_value=500, max_value=100000, value=12000, step=500)
loan_int_rate = st.sidebar.number_input("Interest Rate (%)", min_value=1.0, max_value=40.0, value=11.2, step=0.1)

# Auto-calculate Debt-to-Income ratio or let user tweak
calculated_ratio = round(loan_amnt / max(person_income, 1), 2)
loan_percent_income = st.sidebar.number_input("Loan-to-Income Ratio", min_value=0.0, max_value=1.0, value=min(calculated_ratio, 1.0), step=0.01)

cb_person_default_on_file = st.sidebar.radio("Prior Default on Record?", ["N", "Y"])
cb_person_cred_hist_length = st.sidebar.number_input("Credit History Length (Years)", min_value=0, max_value=50, value=5)

# Build applicant DataFrame
applicant_data = pd.DataFrame([{
    'person_age': person_age,
    'person_income': person_income,
    'person_home_ownership': person_home_ownership,
    'person_emp_length': person_emp_length,
    'loan_intent': loan_intent,
    'loan_grade': loan_grade,
    'loan_amnt': loan_amnt,
    'loan_int_rate': loan_int_rate,
    'loan_percent_income': loan_percent_income,
    'cb_person_default_on_file': cb_person_default_on_file,
    'cb_person_cred_hist_length': cb_person_cred_hist_length
}])

# Main Dashboard Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Loan Assessment Decision")
    if st.button("Evaluate Application", type="primary", use_container_width=True):
        results = predict_loan_risk(applicant_data)
        
        prob = float(results['default_probability'].iloc[0])
        risk_text = results['predicted_risk_level'].iloc[0]

        # Metric Displays
        st.metric("Default Probability", f"{prob * 100:.1f}%")

        if prob >= 0.50:
            st.error(f"**Decision:** DECLINE — {risk_text}")
        else:
            st.success(f"**Decision:** APPROVE — {risk_text}")

        # Risk Gauge Bar
        st.progress(prob)

with col2:
    st.subheader("📊 Model Global Feature Importance")
    try:
        model, preprocessor = load_artifacts()
        
        # Extract features
        cat_cols = preprocessor.transformers_[1][2]
        num_cols = preprocessor.transformers_[0][2]
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
        all_features = num_cols + list(cat_encoder.get_feature_names_out(cat_cols))
        
        importance_df = pd.DataFrame({
            'Feature': all_features,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False).head(8)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax, palette='Blues_r')
        ax.set_title("Top Risk Signals in Model")
        st.pyplot(fig)
    except Exception as e:
        st.info("Train the model first to display global feature importances.")