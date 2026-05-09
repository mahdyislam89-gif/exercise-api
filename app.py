from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib

# =========================
# APP INIT
# =========================
app = FastAPI(title="SmartFit ML API")

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
# LOAD MODELS
# =========================
exercise_model = joblib.load("exercise_pipeline.pkl")
diet_model = joblib.load("diet_pipeline.pkl")
recommendation_model = joblib.load("recomendation_pipeline.pkl")


# =========================
# SCHEMAS
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


class DietData(BaseModel):
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
    Exercises: str


class RecommendationData(BaseModel):
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
    Exercises: str
    Diet: str


# =========================
# HOME
# =========================
@app.get("/")
def home():
    return {
        "message": "SmartFit API Running Successfully 🚀"
    }


# =========================
# EXERCISE PREDICTION
# =========================
@app.post("/predict/exercise")
def predict_exercise(data: ExerciseData):

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

    prediction = exercise_model.predict(test_df)

    return {
        "recommended_exercise": str(prediction[0])
    }


# =========================
# DIET PREDICTION
# =========================
@app.post("/predict/diet")
def predict_diet(data: DietData):

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
        "Equipment": data.Equipment,
        "Exercises": data.Exercises
    }])

    prediction = diet_model.predict(test_df)

    return {
        "recommended_diet": str(prediction[0])
    }


# =========================
# FULL RECOMMENDATION
# =========================
@app.post("/predict/recommendation")
def predict_recommendation(data: RecommendationData):

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
        "Equipment": data.Equipment,
        "Exercises": data.Exercises,
        "Diet": data.Diet
    }])

    prediction = recommendation_model.predict(test_df)

    return {
        "recommendation": str(prediction[0])
    }





"""from fastapi import FastAPI
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
"""