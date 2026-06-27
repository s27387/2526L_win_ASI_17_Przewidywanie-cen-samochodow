import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

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


def _pick_value(cand, col):
    if pd.api.types.is_numeric_dtype(cand[col]):
        return cand[col].median()
    mode_vals = cand[col].mode()
    return mode_vals[0] if not mode_vals.empty else cand.iloc[0][col]


def _impute_by_fingerprint(df, target_cols, core_cols, opt_cols):
    n_core = len(core_cols)
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
            val = _pick_value(cand, col)
            for idx in group.index:
                df.loc[idx, col] = val
    return df


def preprocess_for_xgboost(raw_data: pd.DataFrame, params: dict) -> pd.DataFrame:
    data = raw_data.copy()
    target_column = params.get("target_column", "Price")


    eur_rate = params.get("eur_rate", 4.30)
    eur_mask = data["Currency"] == "EUR"
    data.loc[eur_mask, target_column] = (
        (data.loc[eur_mask, target_column] * eur_rate).round().astype(int)
    )

    data[target_column] = pd.to_numeric(data[target_column], errors="coerce")
    data = data[data[target_column] > 0].reset_index(drop=True)

    q1 = data[target_column].quantile(0.25)
    q3 = data[target_column].quantile(0.75)
    iqr = q3 - q1
    data = data[
        (data[target_column] >= q1 - 1.5 * iqr)
        & (data[target_column] <= q3 + 1.5 * iqr)
    ].reset_index(drop=True)

    data["Production_year"] = pd.to_numeric(data["Production_year"], errors="coerce")
    data = data[data["Production_year"] >= params.get("year_min", 1990)].reset_index(drop=True)

    mileage_cap = params.get("mileage_cap", 450000)
    data["Mileage_km"] = pd.to_numeric(data["Mileage_km"], errors="coerce")
    data = data[~(data["Mileage_km"] < 0)].reset_index(drop=True)
    data.loc[data["Mileage_km"] > mileage_cap, "Mileage_km"] = mileage_cap

    data["_age"] = data["Production_year"].max() - data["Production_year"]
    used_zero = (data["Condition"] == "Used") & (data["Mileage_km"] == 0)
    model_age_mean = (
        data[data["Mileage_km"] > 0]
        .groupby(["Vehicle_model", "_age"])["Mileage_km"]
        .mean()
    )
    idx = pd.MultiIndex.from_frame(data.loc[used_zero, ["Vehicle_model", "_age"]])
    data.loc[used_zero, "Mileage_km"] = idx.map(model_age_mean)
    data = data.drop(columns=["_age"])

    power_cap = params.get("power_cap", 800)
    data["Power_HP"] = pd.to_numeric(data["Power_HP"], errors="coerce")
    data.loc[data["Power_HP"] > power_cap, "Power_HP"] = power_cap

    data["Displacement_cm3"] = pd.to_numeric(data["Displacement_cm3"], errors="coerce")

    keep_fuel = params.get("keep_fuel_types", ["Gasoline", "Diesel", "Gasoline + LPG"])
    data = data[data["Fuel_type"].isin(keep_fuel)].reset_index(drop=True)

    co2_cap_pct = params.get("co2_cap_percentile", 0.99)
    data["CO2_emissions"] = pd.to_numeric(data["CO2_emissions"], errors="coerce")
    for ft in keep_fuel:
        mask = data["Fuel_type"] == ft
        if mask.sum() > 0:
            cap = data.loc[mask, "CO2_emissions"].quantile(co2_cap_pct)
            if not pd.isna(cap):
                data.loc[mask & (data["CO2_emissions"] > cap), "CO2_emissions"] = cap

    doors_min = params.get("doors_min", 2)
    doors_max = params.get("doors_max", 5)
    data["Doors_number"] = pd.to_numeric(data["Doors_number"], errors="coerce")
    data["Doors_number"] = data["Doors_number"].clip(doors_min, doors_max)

    drop_columns = params.get("drop_columns", [
        "Index", "Currency", "Offer_location",
        "Vehicle_version", "Vehicle_generation",
        "First_registration_date", "First_owner",
        "Features", "Origin_country", "CO2_emissions",
    ])
    existing_drop = [c for c in drop_columns if c in data.columns]
    data = data.drop(columns=existing_drop)

    logger.info("Cleaning complete. Shape: %s", data.shape)


    imp_params = params.get("imputation", {})
    core_cols = imp_params.get(
        "core_cols", ["Vehicle_brand", "Vehicle_model", "Production_year", "Type", "Fuel_type"]
    )
    opt_cols = imp_params.get(
        "opt_cols", ["Transmission", "Power_HP", "Doors_number", "Displacement_cm3"]
    )
    target_cols = imp_params.get(
        "target_cols", ["Transmission", "Doors_number", "Power_HP", "Displacement_cm3", "Drive"]
    )
    n_passes = imp_params.get("fingerprint_passes", 2)

    for _ in range(n_passes):
        data = _impute_by_fingerprint(data, target_cols, core_cols, opt_cols)

    num_cols = data.select_dtypes(include=[np.number]).columns
    cat_cols = data.select_dtypes(include=["object"]).columns
    data[num_cols] = SimpleImputer(strategy="median").fit_transform(data[num_cols])
    data[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(data[cat_cols])

    logger.info("Imputation complete. Shape: %s", data.shape)


    pub_dates = pd.to_datetime(data["Offer_publication_date"], errors="coerce")
    pub_years = pub_dates.dt.year.fillna(pd.Timestamp.now().year).astype(int)
    data["car_age"] = (pub_years - data["Production_year"]).clip(lower=0)
    data = data.drop(columns=["Production_year", "Offer_publication_date"])

    data["mileage_per_year"] = data["Mileage_km"] / data["car_age"].clip(lower=1)
    data["power_to_displacement"] = np.where(
        data["Displacement_cm3"] > 0,
        data["Power_HP"] / data["Displacement_cm3"],
        0,
    )
    data["age_x_mileage"] = data["car_age"] * data["Mileage_km"]

    logger.info("Feature engineering complete. Shape: %s", data.shape)
    return data


def analyze_and_select_features(
    data: pd.DataFrame, target_column: str, params: dict, random_state: int
):
    n_top = params.get("n_top_features", 20)
    threshold = params.get("importance_threshold", 0.01)
    n_jobs = params.get("n_jobs", 1)

    X = data.drop(columns=[target_column]).copy()
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype("category").cat.codes
    y = data[target_column]

    rf = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=n_jobs)
    rf.fit(X, y)

    importances = (
        pd.DataFrame({"feature": X.columns, "importance": rf.feature_importances_})
        .sort_values("importance", ascending=False)
    )

    selected = importances[
        (importances["importance"] >= threshold)
        | (importances.reset_index().index < n_top)
    ]
    selected_features = selected["feature"].tolist()

    logger.info(
        "XGBoost feature selection: %d features selected from %d",
        len(selected_features), len(X.columns),
    )
    return selected_features, importances


def split_data(
    data: pd.DataFrame, selected_features: list, target_column: str, params: dict
) -> tuple:
    random_state = params.get("random_state", 42)
    test_size = params.get("test_size", 0.2)

    available = [f for f in selected_features if f in data.columns]
    X = data[available].copy()
    y = data[target_column].values

    for col in X.select_dtypes(include=["object"]).columns:
        converted = pd.to_numeric(X[col], errors="coerce")
        if converted.notna().all():
            X[col] = converted
        else:
            X[col] = X[col].astype("category")

    bins = pd.qcut(y, q=10, labels=False, duplicates="drop")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=bins
    )

    logger.info("XGBoost split: train=%s, test=%s", X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test, available


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    params: dict,
    random_state: int,
    selected_features: list,
) -> tuple:
    init_mlflow("revrate_xgboost")

    mode = params.get("param_search_mode", "quick")
    param_dist_key = f"param_distributions_{mode}"
    param_dist = params.get(param_dist_key, {
        "n_estimators": [200, 400],
        "max_depth": [5, 7, 9],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
    })
    n_iter = params.get("n_iter", 60)
    cv_folds = params.get("cv", 3)
    scoring = params.get("scoring", "neg_root_mean_squared_error")
    n_jobs = params.get("n_jobs", 1)

    base_xgb = XGBRegressor(
        random_state=random_state,
        n_jobs=-1,
        objective="reg:squarederror",
        enable_categorical=True,
        verbosity=0,
        tree_method="hist",
    )

    grid_search = RandomizedSearchCV(
        estimator=base_xgb,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=KFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=2,
        random_state=random_state,
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    logger.info("XGBoost best params: %s", grid_search.best_params_)
    logger.info("XGBoost best CV score: %.4f", grid_search.best_score_)

    with mlflow.start_run(run_name="xgboost_training") as run:
        run_id = run.info.run_id

        mlflow.set_tag("pipeline", "xgboost")
        mlflow.set_tag("n_features", str(X_train.shape[1]))

        mlflow.log_params(grid_search.best_params_)
        mlflow.log_params({
            "cv_folds": cv_folds,
            "n_iter": n_iter,
            "scoring": scoring,
        })

        mlflow.log_metric("best_cv_score", grid_search.best_score_)

        mlflow.log_dict({
            "n_samples": len(X_train),
            "n_features": X_train.shape[1],
            "columns": selected_features,
        }, "dataset_info.json")

    return best_model, run_id


def evaluate_xgboost(
    model: XGBRegressor,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
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
        "train_rmse": train_rmse, "test_rmse": test_rmse,
        "train_mae": train_mae, "test_mae": test_mae,
        "train_r2": train_r2, "test_r2": test_r2,
    }

    logger.info(
        "XGBoost eval: Train RMSE=%.2f, Test RMSE=%.2f, Test R2=%.4f",
        train_rmse, test_rmse, test_r2,
    )

    init_mlflow("revrate_xgboost")
    with mlflow.start_run(run_name="xgboost_evaluation") as run:
        mlflow.set_tag("pipeline", "xgboost")
        mlflow.set_tag("train_run_id", train_run_id)
        mlflow.log_metrics(metrics)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].scatter(y_test, y_test_pred, alpha=0.3, s=2)
        axes[0, 0].plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            "r--", lw=1,
        )
        axes[0, 0].set_xlabel("Actual")
        axes[0, 0].set_ylabel("Predicted")
        axes[0, 0].set_title(f"Test set (R²={test_r2:.3f}, RMSE={test_rmse:.0f})")

        residuals = y_test - y_test_pred
        axes[0, 1].hist(residuals, bins=60, edgecolor="black", alpha=0.7)
        axes[0, 1].axvline(0, color="red", linestyle="--")
        axes[0, 1].set_xlabel("Residual")
        axes[0, 1].set_title(f"Residuals (MAE={test_mae:.0f})")

        top_n = feature_importances.head(20)
        axes[1, 0].barh(top_n["feature"], top_n["importance"])
        axes[1, 0].invert_yaxis()
        axes[1, 0].set_xlabel("Importance")
        axes[1, 0].set_title("Feature importance")

        axes[1, 1].scatter(y_train, y_train_pred, alpha=0.3, s=2)
        axes[1, 1].plot(
            [y_train.min(), y_train.max()],
            [y_train.min(), y_train.max()],
            "r--", lw=1,
        )
        axes[1, 1].set_xlabel("Actual")
        axes[1, 1].set_ylabel("Predicted")
        axes[1, 1].set_title(f"Train set (R²={train_r2:.3f})")

        mlflow.log_figure(fig, "evaluation_plots.png")
        plt.close(fig)

        mlflow.xgboost.log_model(
            model,
            artifact_path="xgboost_model",
            registered_model_name="revrate_xgboost",
        )

    return metrics


def finalize_model(
    model: XGBRegressor,
    feature_list: list,
) -> dict:
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    import joblib
    joblib.dump(model, model_dir / "xgboost_model.pkl")
    joblib.dump(feature_list, model_dir / "xgboost_top_features.pkl")

    logger.info("XGBoost model saved to models/xgboost_model.pkl")
    return {"model_path": str(model_dir / "xgboost_model.pkl")}
