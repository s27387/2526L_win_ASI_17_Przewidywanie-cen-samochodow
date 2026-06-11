import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .. import init_mlflow

logger = logging.getLogger(__name__)


def _find_similar_cars(pool, core_cols, core_vals, opt_cols, opt_vals):
    cand = pool
    for c, v in zip(core_cols, core_vals):
        cand = cand[cand[c] == v]
    opt_present = [c for c, v in zip(opt_cols, opt_vals) if not pd.isna(v)]
    if opt_present:
        cand_opt = cand
        for c, v in zip(opt_cols, opt_vals):
            if not pd.isna(v):
                cand_opt = cand_opt[cand_opt[c] == v]
        if len(cand_opt) > 0:
            cand = cand_opt
    return cand


def _pick_value(cand, col, opt_present):
    if opt_present:
        return cand.iloc[0][col]
    mode_vals = cand[col].mode()
    return mode_vals[0] if not mode_vals.empty else cand.iloc[0][col]


def _impute_pass(df, target_cols, core_cols, opt_cols):
    n_core = len(core_cols)
    records = []
    for col in target_cols:
        missing = df[df[col].isna()]
        if missing.empty:
            continue
        pool = df[df[col].notna()]
        for key, group in missing.groupby(core_cols + opt_cols, dropna=False):
            core_vals = key[:n_core]
            if any(pd.isna(v) for v in core_vals):
                continue
            opt_vals = key[n_core:]
            cand = _find_similar_cars(pool, core_cols, core_vals, opt_cols, opt_vals)
            if len(cand) == 0:
                continue
            opt_present = [c for c, v in zip(opt_cols, opt_vals) if not pd.isna(v)]
            val = _pick_value(cand, col, opt_present)
            for idx in group.index:
                df.loc[idx, col] = val
                records.append({
                    "missing_row": idx,
                    "column": col,
                    "matched_row": cand.index[0],
                    "value": val,
                    "fingerprint_core": str(core_vals),
                    "fingerprint_opt": ",".join(opt_present),
                })
    return records


def clean_data(raw_data: pd.DataFrame, cleaning_params: dict, target_column: str) -> pd.DataFrame:
    data = raw_data.copy()

    eur_rate = cleaning_params.get("eur_rate", 4.30)
    eur_mask = data["Currency"] == "EUR"
    data.loc[eur_mask, target_column] = (data.loc[eur_mask, target_column] * eur_rate).round().astype(int)

    data[target_column] = pd.to_numeric(data[target_column], errors="coerce")
    data = data[data[target_column] > 0].reset_index(drop=True)

    Q1 = data[target_column].quantile(0.25)
    Q3 = data[target_column].quantile(0.75)
    IQR = Q3 - Q1
    data = data[(data[target_column] >= Q1 - 1.5 * IQR) & (data[target_column] <= Q3 + 1.5 * IQR)].reset_index(drop=True)

    year_min = cleaning_params.get("year_min", 1990)
    data["Production_year"] = pd.to_numeric(data["Production_year"], errors="coerce")
    data = data[data["Production_year"] >= year_min].reset_index(drop=True)

    mileage_cap = cleaning_params.get("mileage_cap", 450000)
    data["Mileage_km"] = pd.to_numeric(data["Mileage_km"], errors="coerce")
    data.loc[data["Mileage_km"] > mileage_cap, "Mileage_km"] = mileage_cap
    data = data[data["Mileage_km"] > 0].reset_index(drop=True)

    power_cap = cleaning_params.get("power_cap", 500)
    data["Power_HP"] = pd.to_numeric(data["Power_HP"], errors="coerce")
    data.loc[data["Power_HP"] > power_cap, "Power_HP"] = power_cap

    data["Displacement_cm3"] = pd.to_numeric(data["Displacement_cm3"], errors="coerce")

    keep_fuel = cleaning_params.get("keep_fuel_types", ["Gasoline", "Diesel", "Gasoline + LPG"])
    data = data[data["Fuel_type"].isin(keep_fuel)].reset_index(drop=True)

    co2_cap_pct = cleaning_params.get("co2_cap_percentile", 0.99)
    data["CO2_emissions"] = pd.to_numeric(data["CO2_emissions"], errors="coerce")
    for ft in keep_fuel:
        mask = data["Fuel_type"] == ft
        if mask.sum() > 0:
            cap = data.loc[mask, "CO2_emissions"].quantile(co2_cap_pct)
            if not pd.isna(cap):
                data.loc[mask & (data["CO2_emissions"] > cap), "CO2_emissions"] = cap

    doors_min = cleaning_params.get("doors_min", 2)
    doors_max = cleaning_params.get("doors_max", 5)
    data["Doors_number"] = pd.to_numeric(data["Doors_number"], errors="coerce")
    data["Doors_number"] = data["Doors_number"].clip(doors_min, doors_max)

    drop_columns = cleaning_params.get("drop_columns", [])
    existing_drop = [c for c in drop_columns if c in data.columns]
    data = data.drop(columns=existing_drop)

    logger.info("Data cleaning complete. Shape: %s", data.shape)
    return data


def impute_missing(data: pd.DataFrame, imputation_params: dict) -> pd.DataFrame:
    df = data.copy()

    core_cols = imputation_params.get(
        "core_cols", ["Vehicle_brand", "Vehicle_model", "Production_year", "Type", "Fuel_type"]
    )
    opt_cols = imputation_params.get(
        "opt_cols", ["Transmission", "Power_HP", "Doors_number", "Displacement_cm3"]
    )
    target_cols = imputation_params.get(
        "target_cols",
        ["Transmission", "Doors_number", "Power_HP", "Displacement_cm3", "Drive", "CO2_emissions"],
    )
    n_passes = imputation_params.get("fingerprint_passes", 2)

    matches = []
    for pass_num in range(n_passes):
        pass_start = len(matches)
        records = _impute_pass(df, target_cols, core_cols, opt_cols)
        matches.extend(records)
        logger.info("Pass %d fingerprint: %d", pass_num + 1, len(matches) - pass_start)

    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns
    df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])
    df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

    assert df.isnull().sum().sum() == 0
    logger.info("Imputation complete. Shape: %s", df.shape)
    return df


def engineer_features(data: pd.DataFrame, fe_params: dict, target_column: str) -> pd.DataFrame:
    df = data.copy()
    current_year = fe_params.get("current_year", 2023)

    if fe_params.get("add_voivodeship", True) and "Offer_location" in df.columns:
        def _extract_voivodeship(addr):
            a = str(addr).lower()
            if "mazowieck" in a: return "mazowieckie"
            if "slask" in a: return "slaskie"
            if "wielkopolsk" in a: return "wielkopolskie"
            if "malopolsk" in a: return "malopolskie"
            if "dolnoslask" in a: return "dolnoslaskie"
            if "lodzk" in a: return "lodzkie"
            if "pomorsk" in a: return "pomorskie"
            if "lubelsk" in a: return "lubelskie"
            if "podkarpack" in a: return "podkarpackie"
            if "kujawsko" in a: return "kujawsko-pomorskie"
            if "warminsko" in a: return "warminsko-mazurskie"
            if "zachodniopomorsk" in a: return "zachodniopomorskie"
            if "lubusk" in a: return "lubuskie"
            if "podlask" in a: return "podlaskie"
            if "opolsk" in a: return "opolskie"
            if "swietokrzysk" in a: return "swietokrzyskie"
            return "unknown"

        df["voivodeship"] = df["Offer_location"].apply(_extract_voivodeship)
        df = df.drop(columns=["Offer_location"])

    df["car_age"] = (current_year - df["Production_year"]).clip(lower=0)
    df = df.drop(columns=["Production_year"])

    df["mileage_per_year"] = df["Mileage_km"] / df["car_age"].clip(lower=1)
    df["power_to_displacement"] = np.where(
        df["Displacement_cm3"] > 0,
        df["Power_HP"] / df["Displacement_cm3"],
        0,
    )

    if fe_params.get("add_age_x_mileage", True):
        df["age_x_mileage"] = df["car_age"] * df["Mileage_km"]

    logger.info("Feature engineering complete. Shape: %s", df.shape)
    return df


def analyze_and_select_features(
    data: pd.DataFrame,
    target_column: str,
    selection_params: dict,
    random_state: int,
) -> tuple:
    df = data.copy()

    X_feat = df.drop(columns=[target_column]).copy()
    for col in X_feat.select_dtypes(include=["object"]).columns:
        X_feat[col] = X_feat[col].astype("category").cat.codes
    y_feat = df[target_column]

    rf = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
    rf.fit(X_feat, y_feat)

    importances = pd.DataFrame(
        {"feature": X_feat.columns, "importance": rf.feature_importances_}
    ).sort_values("importance", ascending=False)

    n_top = selection_params.get("n_top_features", 15)
    threshold = selection_params.get("importance_threshold", 0.01)

    top_n = importances.head(n_top)
    above_threshold = importances[importances["importance"] >= threshold]
    selected = pd.concat([top_n, above_threshold]).drop_duplicates(subset="feature")

    selected_features = selected["feature"].tolist()

    logger.info("Selected %d features from %d total.", len(selected_features), len(X_feat.columns))
    return selected_features, importances


def split_data(
    data: pd.DataFrame,
    selected_features: list,
    target_column: str,
    params: dict,
) -> tuple:
    available_features = [f for f in selected_features if f in data.columns]
    X = data[available_features].copy()
    y = data[target_column].values

    test_size = params.get("test_size", 0.2)
    random_state = params.get("random_state", 42)

    bins = pd.qcut(y, q=10, labels=False, duplicates="drop")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=bins
    )

    logger.info("Split complete. Train: %s, Test: %s", X_train_raw.shape, X_test_raw.shape)
    return X_train_raw, X_test_raw, y_train, y_test


def preprocess_data(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    preprocess_params: dict,
) -> tuple:
    numeric_features = X_train_raw.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train_raw.select_dtypes(include=["object"]).columns.tolist()
    max_categories = preprocess_params.get("max_categories", 15)

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            max_categories=max_categories,
            min_frequency=0.001,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    logger.info("Preprocessing complete. Train: %s, Test: %s", X_train.shape, X_test.shape)
    return X_train, X_test, preprocessor


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    rf_params: dict,
    random_state: int,
    selected_features: list,
) -> tuple:
    param_grid = {
        "n_estimators": rf_params.get("n_estimators", [200, 300]),
        "max_depth": rf_params.get("max_depth", [20, 30]),
        "min_samples_leaf": rf_params.get("min_samples_leaf", [1, 4]),
        "max_features": rf_params.get("max_features", ["sqrt"]),
    }

    cv = rf_params.get("cv", 5)
    scoring = rf_params.get("scoring", "neg_root_mean_squared_error")
    n_jobs = rf_params.get("n_jobs", -1)

    base_rf = RandomForestRegressor(random_state=random_state)

    grid_search = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        cv=KFold(n_splits=cv, shuffle=True, random_state=random_state),
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    logger.info("Best params: %s", grid_search.best_params_)
    logger.info("Best CV score: %.4f", grid_search.best_score_)

    init_mlflow("revrate_custom_rf")

    with mlflow.start_run(run_name="custom_rf_training") as run:
        run_id = run.info.run_id

        mlflow.set_tag("pipeline", "custom_rf")
        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("n_features", str(len(selected_features)))

        mlflow.log_params(grid_search.best_params_)
        mlflow.log_params({"cv_folds": cv, "scoring": scoring})
        mlflow.log_metric("best_cv_score", grid_search.best_score_)

        mlflow.log_dict(
            {
                "n_samples": len(X_train),
                "n_features": len(selected_features),
                "features": selected_features,
            },
            "dataset_info.json",
        )

    return best_model, run_id


def evaluate_model(
    model: RandomForestRegressor,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_importances: pd.DataFrame,
    train_run_id: str,
) -> dict:
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    metrics = {
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "train_r2": train_r2,
        "test_r2": test_r2,
    }

    logger.info("Train: RMSE=%.2f, MAE=%.2f, R2=%.4f", train_rmse, train_mae, train_r2)
    logger.info("Test:  RMSE=%.2f, MAE=%.2f, R2=%.4f", test_rmse, test_mae, test_r2)

    init_mlflow("revrate_custom_rf")
    with mlflow.start_run(run_name="custom_rf_evaluation") as run:
        mlflow.set_tag("pipeline", "custom_rf")
        mlflow.set_tag("train_run_id", train_run_id)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "random_forest_model")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.scatter(y_test, y_test_pred, alpha=0.3, s=2)
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=1)
        ax1.set_xlabel("Actual")
        ax1.set_ylabel("Predicted")
        ax1.set_title(f"Test set (R²={test_r2:.3f}, RMSE={test_rmse:.0f})")

        residuals = y_test - y_test_pred
        ax2.hist(residuals, bins=60, edgecolor="black", alpha=0.7)
        ax2.axvline(0, color="red", linestyle="--")
        ax2.set_xlabel("Residual")
        ax2.set_title(f"Residuals (MAE={test_mae:.0f})")

        mlflow.log_figure(fig, "evaluation_plots.png")
        plt.close(fig)

        top_features = feature_importances.head(15).copy()
        top_features = top_features.iloc[::-1]
        fig2, ax = plt.subplots(figsize=(8, 6))
        ax.barh(top_features["feature"], top_features["importance"])
        ax.set_xlabel("Importance")
        mlflow.log_figure(fig2, "feature_importance.png")
        plt.close(fig2)

        model_uri = f"runs:/{run.info.run_id}/random_forest_model"
        mlflow.register_model(model_uri, "revrate_custom_rf")

    return metrics


def finalize_model(
    model: RandomForestRegressor,
    preprocessor: ColumnTransformer,
    selected_features: list,
) -> tuple:
    logger.info("Model, preprocessor and feature list saved to models/")
    return model, preprocessor, selected_features
