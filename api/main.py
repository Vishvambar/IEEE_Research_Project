from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np

# We import the model class
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models'))
try:
    from mtl_model import DAG_MTL_Model
except ImportError:
    pass # Will handle gracefully if missing during early demo

app = FastAPI(title="Clinivora Multi-Disease Early Screening API")

# Allow CORS for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Patient Input Schema
class PatientData(BaseModel):
    age: float
    gender: int # 0=M, 1=F
    heart_rate_avg: float
    sys_bp_avg: float
    dias_bp_avg: float
    mean_bp_avg: float
    resp_rate_avg: float
    temp_avg: float
    spo2_avg: float
    wbc_max: float
    creatinine_max: float
    heart_rate_range: float
    sys_bp_range: float
    resp_rate_range: float
    glucose_max: float
    lactate_max: float
    urineoutput_sum: float

# Load Model
model = None
try:
    model = DAG_MTL_Model(input_dim=17)
    # We would load state_dict here if we had a saved one. 
    # For PoC we will use the initialized model or mock values if un-trained.
    model.eval()
except Exception as e:
    print(f"Model failed to load: {e}")

@app.get("/")
def read_root():
    return {"status": "Clinivora API is running"}

@app.post("/predict")
def predict_risk(patient: PatientData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Feature vector matching our 17 features
    features = torch.tensor([[
        patient.age, patient.gender, patient.heart_rate_avg, patient.sys_bp_avg,
        patient.dias_bp_avg, patient.mean_bp_avg, patient.resp_rate_avg, patient.temp_avg,
        patient.spo2_avg, patient.wbc_max, patient.creatinine_max, patient.heart_rate_range,
        patient.sys_bp_range, patient.resp_rate_range, patient.glucose_max, patient.lactate_max,
        patient.urineoutput_sum
    ]], dtype=torch.float32)

    with torch.no_grad():
        out_tensor = model(features)
        diab, chf, sep, aki = out_tensor[0]
        
    # Convert logits to probabilities
    def sigmoid(x):
        return 1 / (1 + np.exp(-x.item()))
        
    return {
        "Diabetes_Risk": round(sigmoid(diab), 4),
        "CHF_Risk": round(sigmoid(chf), 4),
        "Sepsis_Risk": round(sigmoid(sep), 4),
        "AKI_Risk": round(sigmoid(aki), 4)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
