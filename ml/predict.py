import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.preprocessing import create_engineered_features


def load_artifacts():
    """
    Loads saved preprocessor and model binaries from ml/models/.
    """
    models_dir = os.path.join("ml", "models")
    if not os.path.exists(models_dir):
        models_dir = os.path.join("..", "ml", "models")

    model_path = os.path.join(models_dir, "best_loan_model.joblib")
    preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")

    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        raise FileNotFoundError("Model or Preprocessor artifact missing. Run ml/train.py first.")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor


def plot_feature_importance(model, preprocessor, top_n: int = 10):
    """
    Extracts feature importances from the trained model and plots the top features.
    """
    # Retrieve feature names from ColumnTransformer
    cat_cols = preprocessor.transformers_[1][2]
    num_cols = preprocessor.transformers_[0][2]
    
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
    encoded_cat_features = list(cat_encoder.get_feature_names_out(cat_cols))
    all_feature_names = num_cols + encoded_cat_features

    # Get feature importances from XGBoost / Random Forest
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        print("Model does not support direct feature importances.")
        return

    importance_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(top_n)

    print("\n==================== Top Model Features ====================")
    print(importance_df.to_string(index=False))

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=importance_df, 
        x='Importance', 
        y='Feature', 
        hue='Feature', 
        legend=False, 
        palette='viridis'
    )
    plt.title(f'Top {top_n} Most Important Features for Default Prediction', fontsize=14, fontweight='bold')
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()


def predict_loan_risk(applicant_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes raw applicant input DataFrame, applies feature engineering, transforms
    using saved preprocessor, and returns default probabilities and risk classes.
    """
    model, preprocessor = load_artifacts()

    # Step 1: Feature Engineering
    df_featured = create_engineered_features(applicant_df)

    # Step 2: Transform Features
    X_transformed = preprocessor.transform(df_featured)

    # Step 3: Predict
    probabilities = model.predict_proba(X_transformed)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    # Attach results to input dataframe
    results_df = applicant_df.copy()
    results_df['default_probability'] = np.round(probabilities, 4)
    results_df['predicted_risk_level'] = np.where(predictions == 1, 'HIGH RISK (Default)', 'LOW RISK (Approve)')

    return results_df


if __name__ == "__main__":
    print("--- Loading Model & Plotting Feature Importance ---")
    model, preprocessor = load_artifacts()
    plot_feature_importance(model, preprocessor)

    # Sample test inference on two distinct applicants
    sample_applicants = pd.DataFrame([
        {
            'person_age': 22,
            'person_income': 42000,
            'person_home_ownership': 'RENT',
            'person_emp_length': 1.0,
            'loan_intent': 'PERSONAL',
            'loan_grade': 'D',
            'loan_amnt': 15000,
            'loan_int_rate': 16.0,
            'loan_percent_income': 0.36,
            'cb_person_default_on_file': 'Y',
            'cb_person_cred_hist_length': 2
        },
        {
            'person_age': 35,
            'person_income': 95000,
            'person_home_ownership': 'MORTGAGE',
            'person_emp_length': 8.0,
            'loan_intent': 'VENTURE',
            'loan_grade': 'A',
            'loan_amnt': 10000,
            'loan_int_rate': 7.5,
            'loan_percent_income': 0.11,
            'cb_person_default_on_file': 'N',
            'cb_person_cred_hist_length': 10
        }
    ])

    print("\n--- Running Sample Loan Risk Inference ---")
    predictions = predict_loan_risk(sample_applicants)
    print(predictions[['loan_intent', 'loan_amnt', 'loan_int_rate', 'default_probability', 'predicted_risk_level']])