from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import uvicorn

from ml.predict import predict_loan_risk

app = FastAPI(
    title="Loan Default Risk Prediction API",
    description="Production-grade API for evaluating credit risk using XGBoost",
    version="1.0.0"
)


class LoanApplicant(BaseModel):
    person_age: int = Field(..., ge=18, le=100, json_schema_extra={"example": 25})
    person_income: float = Field(..., gt=0, json_schema_extra={"example": 50000})
    person_home_ownership: str = Field(..., json_schema_extra={"example": "RENT"})
    person_emp_length: float = Field(..., ge=0, json_schema_extra={"example": 3.0})
    loan_intent: str = Field(..., json_schema_extra={"example": "PERSONAL"})
    loan_grade: str = Field(..., json_schema_extra={"example": "B"})
    loan_amnt: float = Field(..., gt=0, json_schema_extra={"example": 10000})
    loan_int_rate: float = Field(..., gt=0, json_schema_extra={"example": 11.5})
    loan_percent_income: float = Field(..., ge=0, le=1, json_schema_extra={"example": 0.20})
    cb_person_default_on_file: str = Field(..., json_schema_extra={"example": "N"})
    cb_person_cred_hist_length: int = Field(..., ge=0, json_schema_extra={"example": 4})


@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "service": "Loan Default Risk Prediction API",
        "version": "1.0.0"
    }


@app.post("/predict")
def predict_risk(applicant: LoanApplicant):
    try:
        # Convert Pydantic model to Single-Row DataFrame
        input_data = pd.DataFrame([applicant.model_dump()])
        
        # Get model prediction
        result_df = predict_loan_risk(input_data)
        
        # Format JSON response
        probability = float(result_df['default_probability'].iloc[0])
        risk_level = str(result_df['predicted_risk_level'].iloc[0])

        return {
            "loan_amount": applicant.loan_amnt,
            "default_probability": probability,
            "risk_assessment": risk_level,
            "decision": "DECLINE" if probability >= 0.50 else "APPROVE"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)