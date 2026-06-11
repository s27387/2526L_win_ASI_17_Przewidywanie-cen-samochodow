import logging

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


def preprocess_for_autogluon(raw_data: pd.DataFrame, target_column: str) -> pd.DataFrame:
    data = raw_data.copy()
    if "Index" in data.columns:
        data = data.drop(columns=["Index"])
    data[target_column] = pd.to_numeric(data[target_column], errors="coerce")
    data = data[data[target_column] > 0].reset_index(drop=True)

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

    train_data = X_train.copy()
    train_data[target_column] = y_train.values

    model_dir = "models/autogluon_models"

    init_mlflow("revrate_autogluon")

    with mlflow.start_run(run_name="autogluon_training") as run:
        run_id = run.info.run_id

        mlflow.set_tag("pipeline", "autogluon")
        mlflow.set_tag("preset", preset)
        mlflow.set_tag("n_features", str(X_train.shape[1]))

        mlflow.log_params({
            "time_limit": time_limit, "preset": preset,
            "problem_type": problem_type, "eval_metric": eval_metric,
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
