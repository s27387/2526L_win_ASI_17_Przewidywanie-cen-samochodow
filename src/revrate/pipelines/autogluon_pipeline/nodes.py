import logging
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .. import init_mlflow

logger = logging.getLogger(__name__)


def _extract_voivodeship(location: object) -> str:
    address = str(location).lower()
    voivodeships = [
        ("mazowieck", "mazowieckie"),
        ("slask", "slaskie"),
        ("wielkopolsk", "wielkopolskie"),
        ("malopolsk", "malopolskie"),
        ("dolnoslask", "dolnoslaskie"),
        ("lodzk", "lodzkie"),
        ("pomorsk", "pomorskie"),
        ("lubelsk", "lubelskie"),
        ("podkarpack", "podkarpackie"),
        ("kujawsko", "kujawsko-pomorskie"),
        ("warminsko", "warminsko-mazurskie"),
        ("zachodniopomorsk", "zachodniopomorskie"),
        ("lubusk", "lubuskie"),
        ("podlask", "podlaskie"),
        ("opolsk", "opolskie"),
        ("swietokrzysk", "swietokrzyskie"),
    ]
    for needle, voivodeship in voivodeships:
        if needle in address:
            return voivodeship
    return "unknown"


def preprocess_for_autogluon(raw_data: pd.DataFrame, params: dict) -> pd.DataFrame:
    data = raw_data.copy()

    target_column = params.get("target_column", "Price")
    eur_rate = params.get("eur_rate", 4.30)

    data[target_column] = pd.to_numeric(data[target_column], errors="coerce").astype(float)
    if "Currency" in data.columns:
        eur_mask = data["Currency"] == "EUR"
        data.loc[eur_mask, target_column] = (
            pd.to_numeric(data.loc[eur_mask, target_column], errors="coerce") * eur_rate
        )

    data = data[data[target_column] > 0].reset_index(drop=True)

    q1 = data[target_column].quantile(0.25)
    q3 = data[target_column].quantile(0.75)
    iqr = q3 - q1
    data = data[
        (data[target_column] >= q1 - 1.5 * iqr)
        & (data[target_column] <= q3 + 1.5 * iqr)
    ].reset_index(drop=True)

    numeric_columns = [
        "Production_year",
        "Mileage_km",
        "Power_HP",
        "Displacement_cm3",
        "CO2_emissions",
        "Doors_number",
    ]
    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    if "Production_year" in data.columns:
        data = data[data["Production_year"] >= params.get("year_min", 1990)].reset_index(drop=True)
    if "Mileage_km" in data.columns:
        data.loc[data["Mileage_km"] > params.get("mileage_cap", 450000), "Mileage_km"] = params.get(
            "mileage_cap", 450000
        )
        data = data[data["Mileage_km"] > 0].reset_index(drop=True)
    if "Power_HP" in data.columns:
        data.loc[data["Power_HP"] > params.get("power_cap", 500), "Power_HP"] = params.get("power_cap", 500)

    keep_fuel = params.get("keep_fuel_types", ["Gasoline", "Diesel", "Gasoline + LPG"])
    if "Fuel_type" in data.columns:
        data = data[data["Fuel_type"].isin(keep_fuel)].reset_index(drop=True)

    if "Offer_location" in data.columns:
        data["voivodeship"] = data["Offer_location"].apply(_extract_voivodeship)

    current_year = params.get("current_year", 2023)
    if "Production_year" in data.columns:
        data["car_age"] = (current_year - data["Production_year"]).clip(lower=0)
    if "Mileage_km" in data.columns and "car_age" in data.columns:
        data["mileage_per_year"] = data["Mileage_km"] / data["car_age"].clip(lower=1)
        data["age_x_mileage"] = data["car_age"] * data["Mileage_km"]
    if "Power_HP" in data.columns and "Displacement_cm3" in data.columns:
        data["power_to_displacement"] = np.where(
            data["Displacement_cm3"] > 0,
            data["Power_HP"] / data["Displacement_cm3"],
            0,
        )

    drop_columns = params.get(
        "drop_columns",
        [
            "Index",
            "Currency",
            "Offer_publication_date",
            "Vehicle_version",
            "Vehicle_generation",
            "First_registration_date",
            "First_owner",
            "Features",
            "Origin_country",
            "Offer_location",
        ],
    )
    existing_drop = [column for column in drop_columns if column in data.columns]
    data = data.drop(columns=existing_drop)
    if params.get("drop_production_year", True) and "Production_year" in data.columns:
        data = data.drop(columns=["Production_year"])

    sample_size = params.get("sample_size")
    if sample_size and len(data) > sample_size:
        data = data.sample(n=sample_size, random_state=params.get("random_state", 42)).reset_index(drop=True)

    logger.info("AutoGluon preprocessed. Shape: %s", data.shape)
    return data


def split_data(
    processed_data: pd.DataFrame, params: dict
) -> tuple:
    target_column = params.get("target_column", "Price")
    test_size = params.get("test_size", 0.2)
    random_state = params.get("random_state", 42)

    y = processed_data[target_column]
    X = processed_data.drop(columns=[target_column])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info("Split: train=%s, test=%s", X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test


def train_autogluon(
    X_train: pd.DataFrame, y_train: pd.Series, params: dict
) -> tuple:
    time_limit = params.get("time_limit", 300)
    preset = params.get("preset", "medium_quality")
    problem_type = params.get("problem_type", "regression")
    eval_metric = params.get("eval_metric", "root_mean_squared_error")
    target_column = params.get("target_column", "Price")
    hyperparameters = params.get("hyperparameters", {"RF": {}, "XT": {}, "KNN": {}})
    num_bag_folds = params.get("num_bag_folds", 0)
    num_stack_levels = params.get("num_stack_levels", 0)
    verbosity = params.get("verbosity", 2)

    train_data = X_train.copy()
    train_data[target_column] = y_train.values

    model_root = Path(params.get("model_dir", "models/autogluon_models"))
    if params.get("use_timestamped_model_dir", True):
        model_path = model_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        model_path = model_root
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_dir = str(model_path)

    init_mlflow("revrate_autogluon")

    with mlflow.start_run(run_name="autogluon_training") as run:
        run_id = run.info.run_id

        mlflow.set_tag("pipeline", "autogluon")
        mlflow.set_tag("preset", preset)
        mlflow.set_tag("n_features", str(X_train.shape[1]))

        mlflow.log_params({
            "time_limit": time_limit, "preset": preset,
            "problem_type": problem_type, "eval_metric": eval_metric,
            "num_bag_folds": num_bag_folds,
            "num_stack_levels": num_stack_levels,
            "model_dir": model_dir,
        })

        mlflow.log_dict({
            "n_samples": len(X_train), "n_features": X_train.shape[1],
            "columns": X_train.columns.tolist(),
        }, "dataset_info.json")

        predictor = TabularPredictor(
            label=target_column,
            path=model_dir,
            problem_type=problem_type,
            eval_metric=eval_metric,
        ).fit(
            train_data=train_data,
            time_limit=time_limit,
            presets=preset,
            hyperparameters=hyperparameters,
            num_bag_folds=num_bag_folds,
            num_stack_levels=num_stack_levels,
            verbosity=verbosity,
        )

        leaderboard = predictor.leaderboard(silent=True)
        logger.info("Leaderboard:\n%s", leaderboard.head())

        mlflow.log_metric("leaderboard_score", leaderboard.iloc[0]["score_val"])
        mlflow.log_text(str(leaderboard), "leaderboard.txt")
        mlflow.log_artifacts(model_dir, artifact_path="autogluon_model")

    return predictor, run_id


def evaluate_autogluon(
    predictor: TabularPredictor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    train_run_id: str,
) -> dict:
    y_pred = predictor.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {"rmse": rmse, "mae": mae, "r2": r2}
    logger.info("Test: RMSE=%.2f, MAE=%.2f, R2=%.4f", rmse, mae, r2)

    init_mlflow("revrate_autogluon")
    with mlflow.start_run(run_name="autogluon_evaluation"):
        mlflow.set_tag("pipeline", "autogluon")
        mlflow.set_tag("train_run_id", train_run_id)
        mlflow.log_metrics(metrics)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.scatter(y_test, y_pred, alpha=0.3, s=2)
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=1)
        ax1.set_xlabel("Actual")
        ax1.set_ylabel("Predicted")
        ax1.set_title(f"Test set (R²={r2:.3f}, RMSE={rmse:.0f})")

        residuals = y_test - y_pred
        ax2.hist(residuals, bins=60, edgecolor="black", alpha=0.7)
        ax2.axvline(0, color="red", linestyle="--")
        ax2.set_xlabel("Residual")
        ax2.set_title(f"Residuals (MAE={mae:.0f})")

        mlflow.log_figure(fig, "evaluation_plots.png")
        plt.close(fig)

    return metrics
