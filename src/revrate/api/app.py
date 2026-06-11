import pickle
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

MODEL_DIR = Path(__file__).parent.parent.parent.parent / "models"

app = FastAPI(title="RevRate", version="0.1.0")

model = None
preprocessor = None
selected_features = None


class CarFeatures(BaseModel):
    Condition: str
    Vehicle_brand: str
    Vehicle_model: str
    Production_year: int = Field(ge=1900, le=2023)
    Mileage_km: float = Field(ge=0)
    Power_HP: float = Field(ge=0)
    Displacement_cm3: float = Field(ge=0)
    Fuel_type: str
    CO2_emissions: Optional[float] = None
    Drive: str
    Transmission: str
    Type: str
    Doors_number: Optional[float] = None
    Colour: str
    voivodeship: Optional[str] = "unknown"


class PredictionResponse(BaseModel):
    predicted_price: float
    currency: str = "PLN"


def load_artifacts():
    global model, preprocessor, selected_features
    try:
        model = joblib.load(MODEL_DIR / "custom_model.pkl")
        preprocessor = joblib.load(MODEL_DIR / "custom_preprocessor.pkl")
        with open(MODEL_DIR / "custom_top_features.pkl", "rb") as f:
            selected_features = pickle.load(f)
    except FileNotFoundError as e:
        raise RuntimeError("Model files not found. Run `kedro run` first.") from e


@app.on_event("startup")
def startup_event():
    load_artifacts()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CarFeatures):
    global model, preprocessor, selected_features
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    current_year = 2023
    car_age = max(current_year - features.Production_year, 0)
    mileage_per_year = features.Mileage_km / max(car_age, 1)
    power_to_displacement = (
        features.Power_HP / features.Displacement_cm3
        if features.Displacement_cm3 > 0
        else 0
    )
    age_x_mileage = car_age * features.Mileage_km

    input_data = {
        "Condition": features.Condition,
        "Vehicle_brand": features.Vehicle_brand,
        "Vehicle_model": features.Vehicle_model,
        "Mileage_km": features.Mileage_km,
        "Power_HP": features.Power_HP,
        "Displacement_cm3": features.Displacement_cm3,
        "Fuel_type": features.Fuel_type,
        "CO2_emissions": features.CO2_emissions if features.CO2_emissions is not None else np.nan,
        "Drive": features.Drive,
        "Transmission": features.Transmission,
        "Type": features.Type,
        "Doors_number": features.Doors_number if features.Doors_number is not None else np.nan,
        "Colour": features.Colour,
        "voivodeship": features.voivodeship if features.voivodeship else "unknown",
        "car_age": car_age,
        "mileage_per_year": mileage_per_year,
        "power_to_displacement": power_to_displacement,
        "age_x_mileage": age_x_mileage,
    }

    df = pd.DataFrame([input_data])

    available_features = [f for f in selected_features if f in df.columns]
    df = df[available_features]

    df_preprocessed = preprocessor.transform(df)
    prediction = model.predict(df_preprocessed)[0]

    return PredictionResponse(predicted_price=round(float(prediction), 2))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
