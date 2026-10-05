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

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

from ml.preprocessing import prepare_data


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    metrics = {
        'Model': model_name,
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall (Default)': round(rec, 4),
        'F1-Score': round(f1, 4),
        'ROC-AUC': round(roc_auc, 4)
    }

    print(f"\n==================== {model_name} Evaluation ====================")
    print(classification_report(y_test, y_pred, target_names=['Non-Default', 'Default']))
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    
    return metrics


def run_model_experiments(data_path: str):
    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data(data_path)

    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_pos_weight = num_neg / num_pos

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.05, 
            scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss'
        )
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)
        trained_models[name] = model

    comparison_df = pd.DataFrame(results).sort_values(by='ROC-AUC', ascending=False)
    print("\n==================== Model Comparison Matrix ====================")
    print(comparison_df.to_string(index=False))

    best_model_name = comparison_df.iloc[0]['Model']
    best_model = trained_models[best_model_name]
    print(f"\n---> Winning Model Selected: {best_model_name}")

    models_dir = os.path.join("ml", "models")
    os.makedirs(models_dir, exist_ok=True)
    best_model_path = os.path.join(models_dir, "best_loan_model.joblib")
    
    joblib.dump(best_model, best_model_path)
    print(f"Saved winning model to: {best_model_path}")

    return comparison_df, best_model, X_test, y_test, feature_names


if __name__ == "__main__":
    DATA_PATH = os.path.join("data", "processed", "clean_credit_risk.csv")
    if not os.path.exists(DATA_PATH):
        DATA_PATH = os.path.join("..", "data", "processed", "clean_credit_risk.csv")

    run_model_experiments(DATA_PATH)
    