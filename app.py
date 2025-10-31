import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import logging

# ------------------------------
# Configuration
# ------------------------------
MODEL_PATH = "salary_model_lgb.pkl"
ENCODER_PATH = "encoders_lgb.pkl"
DATA_PATH = "clean_salary_dataset.csv"
API_KEY = os.getenv("API_KEY", None)  # optional
ALLOWED_ORIGINS = ["*"]

# ------------------------------
# Logging setup
# ------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smartpay-api")

# ------------------------------
# Load model + encoders
# ------------------------------
try:
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODER_PATH)
    logger.info("✅ Model and encoders loaded successfully.")
except Exception as e:
    logger.exception("Failed to load model/encoders: %s", e)
    raise RuntimeError("Model loading failed")

# ------------------------------
# FastAPI app
# ------------------------------
app = FastAPI(title="SmartPay Salary API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# Request & Response Models
# ------------------------------
class PredictRequest(BaseModel):
    age: int = Field(..., ge=16, le=100)
    education: str
    job_title: str
    hours_per_week: int = Field(..., ge=1, le=100)
    gender: str
    marital_status: str

class PredictResponse(BaseModel):
    predicted_salary_usd: float

# ------------------------------
# Auth dependency (optional)
# ------------------------------
def api_key_auth(x_api_key: Optional[str] = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True

# ------------------------------
# Predict Endpoint
# ------------------------------
@app.post("/predict", response_model=PredictResponse)
def predict_salary(req: PredictRequest, auth: bool = Depends(api_key_auth)):
    try:
        feat_order = ['age', 'education', 'job_title', 'hours_per_week', 'gender', 'marital_status']
        vals = []
        d = req.dict()

        for f in feat_order:
            if f in ['education', 'job_title', 'gender', 'marital_status']:
                enc = encoders.get(f)
                if not enc:
                    vals.append(-1)
                else:
                    try:
                        vals.append(int(enc.transform([str(d[f])])[0]))
                    except Exception:
                        vals.append(-1)
            else:
                vals.append(float(d[f]))

        X = np.array([vals])
        pred = model.predict(X)[0]
        return PredictResponse(predicted_salary_usd=float(pred))
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.get("/")
def root():
    return {"service": "SmartPay Salary Prediction API", "status": "running"}

