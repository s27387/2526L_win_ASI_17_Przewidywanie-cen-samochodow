"""RevRate - Streamlit UI for car price prediction."""

import json
from pathlib import Path

import requests
import streamlit as st

FRONTEND_DIR = Path(__file__).resolve().parent

with open(FRONTEND_DIR / "car_options.json", encoding="utf-8") as f:
    OPTIONS = json.load(f)

with open(FRONTEND_DIR / "labels.json", encoding="utf-8") as f:
    LABELS = json.load(f)

LABELS_TO_RAW = {v: k for mapping in LABELS.values() for k, v in mapping.items()}

MIDDLEWARE_URL = "http://localhost:8001/predict"

HIERARCHY = OPTIONS["hierarchy"]
TRANSMISSIONS = OPTIONS["transmissions"]
FUEL_TYPES = OPTIONS["fuel_types"]
DRIVES = OPTIONS["drives"]
COLOURS = OPTIONS["colours"]
DOORS_NUMBERS = OPTIONS["doors_numbers"]
CONDITIONS = OPTIONS["conditions"]
YEARS = list(range(2026, 1989, -1))

PH = "---"

RESET_DEFAULTS = {
    "brand": PH, "model": PH, "type_label": PH,
    "production_year": PH, "drive_label": PH, "fuel_label": PH,
    "colour_label": PH, "condition_label": PH, "doors": PH,
    "transmission": PH, "power_hp": 150, "mileage_km": 120000,
    "displacement": 2000,
}


def reset_form():
    """Reset all widget state to constructor defaults and clear result."""
    for key, default in RESET_DEFAULTS.items():
        st.session_state[key] = default
    st.session_state.pop("price", None)
    st.session_state.pop("price_pln", None)


def pretty(label_or_raw, category):
    """Convert raw value to pretty label using labels.json."""
    mapping = LABELS.get(category, {})
    return mapping.get(label_or_raw, label_or_raw)


def get_raw(label):
    """Convert pretty label back to raw API value."""
    return LABELS_TO_RAW.get(label, label)


st.set_page_config(page_title="RevRate", page_icon="🚗")

st.markdown(
    """
    <style>
    .big-title h1 {
        background: linear-gradient(135deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        margin-bottom: 0;
    }
    .subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    div.stButton > button {
        font-size: 1rem;
        font-weight: 700;
        padding: 0.75rem 2rem;
        border: none;
        border-radius: 8px;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #38bdf8, #818cf8);
        color: #0f172a;
    }
    div.stButton > button[kind="primary"]:hover {
        opacity: 0.9;
    }
    .result-box {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1rem;
    }
    .result-success {
        border-color: #22c55e;
    }
    .price-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .range-value {
        font-size: 1rem;
        color: #94a3b8;
    }
    .result-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="big-title"><h1>RevRate</h1></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">'
    'Sprawdz przewidywana cene samochodu</div>',
    unsafe_allow_html=True,
)

brands = sorted(HIERARCHY.keys())

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Marka *", [PH] + brands, index=0, key="brand")

models_list = sorted(HIERARCHY.get(brand, {}).keys()) if brand != PH else []
with col2:
    model = st.selectbox(
        "Model *", [PH] + models_list, index=0, key="model",
        disabled=brand == PH,
    )

types_list = (
    sorted(HIERARCHY.get(brand, {}).get(model, []))
    if PH not in (brand, model)
    else []
)
col3, col4 = st.columns(2)

with col3:
    type_label = st.selectbox(
        "Typ nadwozia *", [PH] + types_list, index=0,
        key="type_label", disabled=model == PH,
    )

all_drives = [pretty(d, "drive") for d in DRIVES]
with col4:
    drive_label = st.selectbox("Naped *", [PH] + all_drives, index=0, key="drive_label")

col5, col6, col7 = st.columns(3)

with col5:
    production_year = st.selectbox(
        "Rok produkcji *", [PH] + YEARS, index=0,
        key="production_year",
    )

with col6:
    power_hp = st.number_input("Moc (KM) *", min_value=1, value=150, step=1, key="power_hp")

with col7:
    mileage_km = st.number_input(
        "Przebieg (km) *", min_value=0, value=120000,
        step=1000, key="mileage_km",
    )

col8, col9, col10 = st.columns(3)

with col8:
    displacement = st.number_input(
        "Pojemnosc (cm3) *", min_value=1, value=2000,
        step=1, key="displacement",
    )

with col9:
    doors = st.selectbox(
        "Liczba drzwi *", [PH] + [str(d) for d in DOORS_NUMBERS], index=0, key="doors",
    )

all_transmissions = TRANSMISSIONS[:]
with col10:
    transmission = st.selectbox(
        "Skrzynia biegow *", [PH] + all_transmissions, index=0, key="transmission",
    )

col11, col12, col13 = st.columns(3)

all_fuels = [pretty(f, "fuel") for f in FUEL_TYPES]
with col11:
    fuel_label = st.selectbox("Typ paliwa *", [PH] + all_fuels, index=0, key="fuel_label")

all_colours = [pretty(c, "colour") for c in COLOURS]
with col12:
    colour_label = st.selectbox("Kolor *", [PH] + all_colours, index=0, key="colour_label")

all_conditions = [pretty(c, "condition") for c in CONDITIONS]
with col13:
    condition_label = st.selectbox("Stan *", [PH] + all_conditions, index=0, key="condition_label")

all_filled = all(
    [
        brand != PH,
        model != PH,
        type_label != PH,
        production_year != PH,
        drive_label != PH,
        fuel_label != PH,
        transmission != PH,
        colour_label != PH,
        condition_label != PH,
        doors != PH,
    ]
)

btn_col1, btn_col2 = st.columns(2)

with btn_col1:
    predict_clicked = st.button(
        "Przewidz cene", disabled=not all_filled, type="primary",
        use_container_width=True,
    )

with btn_col2:
    st.button("Wyczysc", use_container_width=True, on_click=reset_form)

if predict_clicked:
    payload = {
        "Production_year": int(production_year),
        "Power_HP": float(power_hp),
        "Mileage_km": float(mileage_km),
        "Displacement_cm3": float(displacement),
        "Doors_number": int(doors),
        "Transmission": transmission,
        "Vehicle_model": model,
        "Vehicle_brand": brand,
        "Type": get_raw(type_label),
        "Drive": get_raw(drive_label),
        "Colour": get_raw(colour_label),
        "Fuel_type": get_raw(fuel_label),
        "Condition": get_raw(condition_label),
    }

    try:
        resp = requests.post(MIDDLEWARE_URL, json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        st.session_state["price"] = result["price_formatted"]
        st.session_state["price_pln"] = result["price_pln"]
    except requests.exceptions.ConnectionError:
        st.error(
            "Nie mozna polaczyc z serwerem. "
            "Upewnij sie, ze aplikacja dziala na porcie 8001."
        )
    except Exception as exc:
        st.error(f"Blad: {exc}")

if "price_pln" in st.session_state:
    price_pln = st.session_state["price_pln"]
    price_fmt = st.session_state["price"]
    lower = int(round(price_pln * 0.85))
    upper = int(round(price_pln * 1.15))
    lower_fmt = f"{lower:,} PLN".replace(",", " ")
    upper_fmt = f"{upper:,} PLN".replace(",", " ")

    st.markdown(
        f"""
        <div class="result-box result-success">
            <div class="result-label">Szacowana cena</div>
            <div class="price-value">{price_fmt}</div>
            <div class="range-value">
                Zakres orientacyjny:<br>
                <strong>{lower_fmt}</strong> &mdash; <strong>{upper_fmt}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
