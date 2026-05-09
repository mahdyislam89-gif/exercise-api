from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# =========================
# APP INIT
# =========================
app = FastAPI()

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# LOAD MODEL
# =========================
model = joblib.load("exercise_pipeline.pkl")


# =========================
# INPUT SCHEMA
# =========================
class ExerciseData(BaseModel):
    Sex: str
    Age: int
    Height: float
    Weight: float
    Hypertension: str
    Diabetes: str
    BMI: float
    Level: str
    Fitness_Goal: str
    Fitness_Type: str
    Equipment: str


# =========================
# HOME
# =========================
@app.get("/")
def home():
    return {"message": "Exercise Recommendation API Running 🚀"}


# =========================
# PREDICT
# =========================
@app.post("/predict")
def predict(data: ExerciseData):

    test_df = pd.DataFrame([{
        "Sex": data.Sex,
        "Age": data.Age,
        "Height": data.Height,
        "Weight": data.Weight,
        "Hypertension": data.Hypertension,
        "Diabetes": data.Diabetes,
        "BMI": data.BMI,
        "Level": data.Level,
        "Fitness Goal": data.Fitness_Goal,
        "Fitness Type": data.Fitness_Type,
        "Equipment": data.Equipment
    }])

    prediction = model.predict(test_df)

    return {
        "recommended_exercise": str(prediction[0])
    }
