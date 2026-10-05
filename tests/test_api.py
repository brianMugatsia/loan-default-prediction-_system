import sys
import os
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app import app

client = TestClient(app)


def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_low_risk_prediction():
    payload = {
        "person_age": 35,
        "person_income": 95000,
        "person_home_ownership": "MORTGAGE",
        "person_emp_length": 8.0,
        "loan_intent": "VENTURE",
        "loan_grade": "A",
        "loan_amnt": 10000,
        "loan_int_rate": 7.5,
        "loan_percent_income": 0.11,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 10
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "default_probability" in data
    assert data["decision"] == "APPROVE"


def test_high_risk_prediction():
    payload = {
        "person_age": 22,
        "person_income": 42000,
        "person_home_ownership": "RENT",
        "person_emp_length": 1.0,
        "loan_intent": "PERSONAL",
        "loan_grade": "D",
        "loan_amnt": 15000,
        "loan_int_rate": 16.0,
        "loan_percent_income": 0.36,
        "cb_person_default_on_file": "Y",
        "cb_person_cred_hist_length": 2
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "default_probability" in data
    assert data["decision"] == "DECLINE"