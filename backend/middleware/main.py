"""Car Price Prediction Middleware - serves frontend and proxies to API."""

# pylint: disable=import-error
import json
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException

from pydantic import BaseModel, validator

API_URL = "http://localhost:8000/predict"
FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"
OPTIONS_PATH = FRONTEND_DIR / "car_options.json"

app = FastAPI(title="Car Price Prediction - Middleware")


class FormData(BaseModel):
    """Form data schema for car price prediction request."""

    Production_year: int
    Power_HP: float
    Mileage_km: float
    Displacement_cm3: float
    Doors_number: int
    Transmission: str
    Vehicle_model: str
    Vehicle_brand: str
    Type: str
    Drive: str
    Colour: str
    Fuel_type: str
    Condition: str

    @validator("Condition")
    def validate_condition(cls, v):
        """Validate that Condition is New or Used."""
        if v not in ("New", "Used"):
            raise ValueError("Condition must be New or Used")
        return v

    @validator("Production_year")
    def validate_production_year(cls, v):
        """Validate that Production_year is between 1990 and 2026."""
        if v < 1990 or v > 2026:
            raise ValueError(
                "Production_year must be between 1990 and 2026"
            )
        return v

    @validator("Power_HP", "Mileage_km", "Displacement_cm3")
    def validate_positive(cls, v):
        """Validate that numeric fields are non-negative."""
        if v < 0:
            raise ValueError("Value must be non-negative")
        return v

    @validator("Doors_number")
    def validate_doors(cls, v):
        """Validate that Doors_number is 2, 3, 4, or 5."""
        if v not in (2, 3, 4, 5):
            raise ValueError("Doors_number must be 2, 3, 4, or 5")
        return v

    class Config:
        """Pydantic config to forbid extra fields."""

        extra = "forbid"


class PriceResponse(BaseModel):
    """Response schema for predicted price."""

    price_pln: int
    price_formatted: str


def compute_derived_features(data: FormData) -> dict:
    """Compute derived features from form data for model input."""
    current_year = datetime.now().year
    car_age = max(current_year - data.Production_year, 0)
    mileage_per_year = data.Mileage_km / max(car_age, 1)
    power_to_displacement = (
        data.Power_HP / data.Displacement_cm3
        if data.Displacement_cm3 > 0
        else 0.0
    )
    age_x_mileage = car_age * data.Mileage_km

    return {
        "Condition": data.Condition,
        "Vehicle_brand": data.Vehicle_brand,
        "Vehicle_model": str(data.Vehicle_model),
        "Mileage_km": data.Mileage_km,
        "Power_HP": data.Power_HP,
        "Displacement_cm3": data.Displacement_cm3,
        "Fuel_type": data.Fuel_type,
        "Drive": data.Drive,
        "Transmission": data.Transmission,
        "Type": data.Type,
        "Doors_number": float(data.Doors_number),
        "Colour": data.Colour,
        "car_age": float(car_age),
        "mileage_per_year": mileage_per_year,
        "power_to_displacement": power_to_displacement,
        "age_x_mileage": age_x_mileage,
    }


@app.get("/car-options")
def get_car_options():
    """Return car option choices from pre-generated JSON file."""
    with open(OPTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/predict", response_model=PriceResponse)
async def predict(data: FormData):
    """Proxy prediction request to the model API."""
    features = compute_derived_features(data)

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(API_URL, json=features)
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise HTTPException(
                status_code=502,
                detail=(
                    "Nie mozna polaczyc z API modelu. "
                    "Upewnij sie, ze api dziala na porcie 8000."
                ),
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=exc.response.status_code,
                detail=f"Blad API: {exc.response.text}",
            ) from exc

    result = response.json()
    price = result["predicted_price"]

    return PriceResponse(
        price_pln=int(round(price)),
        price_formatted=f"{int(round(price)):,} PLN".replace(",", " "),
    )


@app.get("/")
async def root():
    """Redirect root to the Streamlit frontend."""
    from fastapi.responses import RedirectResponse  # pylint: disable=import-outside-toplevel

    return RedirectResponse(url="http://localhost:8501")
