from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn
import os

app = FastAPI(title="Student Dropout Prediction API")

MODEL_PATH = "artifacts/model.joblib"
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

class PredictionRequest(BaseModel):
    age_at_enrollment: float
    gender: int
    admission_grade_pct: float
    scholarship_holder: int
    debtor: int
    displaced: int
    previous_semester_enrolled_subjects: int
    previous_semester_approved_subjects: int
    previous_semester_avg_grade_pct: float

class PredictionResponse(BaseModel):
    dropout_probability: float
    prediction: int
    label: str

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
        else:
            raise HTTPException(status_code=503, detail="Model is not available yet. Please train the model first.")
            
    if request.previous_semester_enrolled_subjects > 0:
        pass_ratio = (request.previous_semester_approved_subjects / request.previous_semester_enrolled_subjects) * 100.0
    else:
        pass_ratio = 0.0

    df = pd.DataFrame([{
        "age_at_enrollment": request.age_at_enrollment,
        "gender": request.gender,
        "admission_grade_pct": request.admission_grade_pct,
        "scholarship_holder": request.scholarship_holder,
        "debtor": request.debtor,
        "displaced": request.displaced,
        "previous_semester_pass_ratio_pct": pass_ratio,
        "previous_semester_avg_grade_pct": request.previous_semester_avg_grade_pct
    }])
    
    prob = model.predict_proba(df)[0, 1]
    pred = model.predict(df)[0]
    label = "At Risk of Dropout" if pred == 1 else "Low Risk"
    
    return PredictionResponse(
        dropout_probability=float(prob),
        prediction=int(pred),
        label=label
    )

app.mount("/", StaticFiles(directory="api/static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
