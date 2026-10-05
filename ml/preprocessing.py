import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates new domain-specific financial features.
    """
    data = df.copy()

    # 1. Credit history length relative to applicant age
    data['cred_hist_to_age_ratio'] = data['cb_person_cred_hist_length'] / (data['person_age'] + 1e-5)

    # 2. Employment length relative to age
    data['emp_length_to_age_ratio'] = data['person_emp_length'] / (data['person_age'] + 1e-5)

    # 3. Estimated annual interest burden ($)
    data['annual_interest_burden'] = data['loan_amnt'] * (data['loan_int_rate'] / 100.0)

    # 4. Binary flag for high-risk loan intent
    data['is_high_risk_intent'] = data['loan_intent'].isin(['MEDICAL', 'DEBTCONSOLIDATION']).astype(int)

    return data


def build_preprocessor(categorical_cols: list, numerical_cols: list) -> ColumnTransformer:
    """
    Builds a Scikit-Learn ColumnTransformer pipeline.
    """
    num_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numerical_cols),
            ('cat', cat_pipeline, categorical_cols)
        ]
    )

    return preprocessor


def prepare_data(data_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    Loads clean data, engineers features, executes stratified train-test split, 
    and fits the preprocessor without data leakage.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Clean dataset not found at {data_path}")

    df = pd.read_csv(data_path)

    # Step 1: Feature Engineering
    df_featured = create_engineered_features(df)

    # Step 2: Separate Features (X) and Target (y)
    X = df_featured.drop(columns=['loan_status'])
    y = df_featured['loan_status']

    # Identify numerical and categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Step 3: Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Step 4: Fit Preprocessor on X_train ONLY
    preprocessor = build_preprocessor(categorical_cols, numerical_cols)
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Retrieve feature names after One-Hot Encoding
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
    encoded_cat_features = list(cat_encoder.get_feature_names_out(categorical_cols))
    all_feature_names = numerical_cols + encoded_cat_features

    # Convert back to DataFrame
    X_train_df = pd.DataFrame(X_train_transformed, columns=all_feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_transformed, columns=all_feature_names, index=X_test.index)

    # Step 5: Save Fitted Preprocessor artifact
    models_dir = os.path.join("ml", "models")
    os.makedirs(models_dir, exist_ok=True)
    preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)

    print(f"--- Data Preparation Complete ---")
    print(f"Training shape: {X_train_df.shape}")
    print(f"Testing shape:  {X_test_df.shape}")
    print(f"Saved fitted preprocessor to: {preprocessor_path}")

    return X_train_df, X_test_df, y_train, y_test, preprocessor, all_feature_names


if __name__ == "__main__":
    DATA_PATH = os.path.join("data", "processed", "clean_credit_risk.csv")
    if not os.path.exists(DATA_PATH):
        DATA_PATH = os.path.join("..", "data", "processed", "clean_credit_risk.csv")

    prepare_data(DATA_PATH)