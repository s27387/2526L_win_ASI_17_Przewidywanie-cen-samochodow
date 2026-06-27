"""Car Price Prediction API endpoint."""

# pylint: disable=import-error
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

MODEL = None
CATEGORICAL_COLS = None
FEATURE_ORDER = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load model and feature metadata on startup."""
    # pylint: disable-next=global-statement
    global MODEL, CATEGORICAL_COLS, FEATURE_ORDER
    MODEL = joblib.load(MODELS_DIR / "custom_model.pkl")
    booster = MODEL.get_booster()
    FEATURE_ORDER = booster.feature_names
    CATEGORICAL_COLS = [
        name for name, ftype in zip(booster.feature_names, booster.feature_types)
        if ftype == "c"
    ]
    yield


app = FastAPI(title="Car Price Prediction API", lifespan=lifespan)


class PredictionRequest(BaseModel):
    """Request schema for car price prediction."""

    Condition: Optional[str] = None
    Vehicle_brand: Optional[str] = None
    Vehicle_model: Optional[str] = None
    Mileage_km: Optional[float] = None
    Power_HP: Optional[float] = None
    Displacement_cm3: Optional[float] = None
    Fuel_type: Optional[str] = None
    Drive: Optional[str] = None
    Transmission: Optional[str] = None
    Type: Optional[str] = None
    Doors_number: Optional[float] = None
    Colour: Optional[str] = None
    car_age: Optional[float] = None
    mileage_per_year: Optional[float] = None
    power_to_displacement: Optional[float] = None
    age_x_mileage: Optional[float] = None


class PredictionResponse(BaseModel):
    """Response schema for car price prediction."""

    predicted_price: float
    currency: str = "PLN"


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Predict car price based on request features."""
    row = {}
    for field in request.model_fields:
        val = getattr(request, field)
        if val is None:
            raise HTTPException(status_code=400, detail=f"Missing feature: {field}")
        row[field] = val

    df = pd.DataFrame([row], columns=FEATURE_ORDER)

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    y_pred = MODEL.predict(df)
    return PredictionResponse(
        predicted_price=float(y_pred[0]),
        currency="PLN",
    )


@app.get("/health")
def health():
    """Return API health status and model metadata."""
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "feature_order": FEATURE_ORDER,
        "categorical_cols": CATEGORICAL_COLS,
    }
