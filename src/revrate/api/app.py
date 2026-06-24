import csv
import json
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

BASE_DIR = Path(__file__).parent.parent.parent.parent
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"
PREDICTION_LOG_PATH = LOG_DIR / "predictions.csv"
REFERENCE_STATS_PATH = MODEL_DIR / "reference_stats.json"

app = FastAPI(title="RevRate", version="0.1.0")

model = None
preprocessor = None
selected_features = None
reference_stats = None


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
    drift_detected: bool = False
    drift_warnings: list[str] = Field(default_factory=list)


def load_artifacts():
    global model, preprocessor, selected_features, reference_stats
    try:
        model = joblib.load(MODEL_DIR / "custom_model.pkl")
        preprocessor = joblib.load(MODEL_DIR / "custom_preprocessor.pkl")
        with open(MODEL_DIR / "custom_top_features.pkl", "rb") as f:
            selected_features = pickle.load(f)
    except FileNotFoundError as e:
        raise RuntimeError("Model files not found. Run `kedro run` first.") from e

    reference_stats = load_reference_stats()


def load_reference_stats() -> dict | None:
    if not REFERENCE_STATS_PATH.exists():
        return None
    with open(REFERENCE_STATS_PATH, encoding="utf-8") as f:
        return json.load(f)


def _features_to_dict(features: CarFeatures) -> dict:
    if hasattr(features, "model_dump"):
        return features.model_dump()
    return features.dict()


def _build_model_input(features: CarFeatures) -> dict:
    current_year = 2023
    car_age = max(current_year - features.Production_year, 0)
    mileage_per_year = features.Mileage_km / max(car_age, 1)
    power_to_displacement = (
        features.Power_HP / features.Displacement_cm3
        if features.Displacement_cm3 > 0
        else 0
    )
    age_x_mileage = car_age * features.Mileage_km

    return {
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


def detect_drift(input_data: dict) -> list[str]:
    stats = reference_stats or load_reference_stats()
    if not stats:
        return ["reference_stats_missing"]

    warnings = []
    for column, column_stats in stats.get("numeric", {}).items():
        value = input_data.get(column)
        if value is None or pd.isna(value):
            continue

        lower = column_stats.get("p01")
        upper = column_stats.get("p99")
        if lower is None or upper is None:
            continue
        if value < lower or value > upper:
            warnings.append(
                f"{column}={value} outside reference p01-p99 range [{lower:.2f}, {upper:.2f}]"
            )

    for column, column_stats in stats.get("categorical", {}).items():
        value = input_data.get(column)
        if value is None or pd.isna(value):
            continue

        known_values = set(column_stats.get("known_values", []))
        if str(value) not in known_values:
            warnings.append(f"{column}={value} not seen in training data")

    return warnings


def canonicalize_categories(input_data: dict) -> dict:
    stats = reference_stats or load_reference_stats()
    if not stats:
        return input_data

    canonicalized = input_data.copy()
    for column, column_stats in stats.get("categorical", {}).items():
        value = canonicalized.get(column)
        if value is None or pd.isna(value):
            continue

        known_values = column_stats.get("known_values", [])
        canonical_by_lower = {str(known).lower(): known for known in known_values}
        canonical_value = canonical_by_lower.get(str(value).lower())
        if canonical_value is not None:
            canonicalized[column] = canonical_value

    return canonicalized


def log_prediction(
    features: CarFeatures,
    predicted_price: float,
    latency_ms: float,
    drift_warnings: list[str],
) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = PREDICTION_LOG_PATH.exists()
    fieldnames = [
        "timestamp",
        "predicted_price",
        "latency_ms",
        "drift_detected",
        "drift_warnings",
        "input_json",
    ]

    with open(PREDICTION_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "predicted_price": round(float(predicted_price), 2),
                "latency_ms": round(float(latency_ms), 2),
                "drift_detected": bool(drift_warnings),
                "drift_warnings": json.dumps(drift_warnings, ensure_ascii=False),
                "input_json": json.dumps(_features_to_dict(features), ensure_ascii=False),
            }
        )


def read_prediction_log() -> list[dict]:
    if not PREDICTION_LOG_PATH.exists():
        return []
    with open(PREDICTION_LOG_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "reference_stats_available": REFERENCE_STATS_PATH.exists(),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CarFeatures):
    global model, preprocessor, selected_features
    start_time = time.perf_counter()

    if model is None or preprocessor is None or selected_features is None:
        try:
            load_artifacts()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    input_data = canonicalize_categories(_build_model_input(features))
    drift_warnings = detect_drift(input_data)

    df = pd.DataFrame([input_data])

    available_features = [f for f in selected_features if f in df.columns]
    df = df[available_features]

    df_preprocessed = preprocessor.transform(df)
    prediction = model.predict(df_preprocessed)[0]
    predicted_price = round(float(prediction), 2)
    latency_ms = (time.perf_counter() - start_time) * 1000
    log_prediction(features, predicted_price, latency_ms, drift_warnings)

    return PredictionResponse(
        predicted_price=predicted_price,
        drift_detected=bool(drift_warnings),
        drift_warnings=drift_warnings,
    )


@app.get("/monitoring/summary")
def monitoring_summary():
    rows = read_prediction_log()
    total_predictions = len(rows)
    drifted_predictions = sum(row.get("drift_detected") == "True" for row in rows)
    last_prediction_at = rows[-1]["timestamp"] if rows else None

    return {
        "total_predictions": total_predictions,
        "drifted_predictions": drifted_predictions,
        "drift_rate": drifted_predictions / total_predictions if total_predictions else 0,
        "last_prediction_at": last_prediction_at,
        "log_path": str(PREDICTION_LOG_PATH),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
