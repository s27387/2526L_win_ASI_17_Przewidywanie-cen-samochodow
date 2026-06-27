from kedro.pipeline import Pipeline, node

from ..custom_pipeline.nodes import download_raw_data

from .nodes import (
    preprocess_for_xgboost,
    analyze_and_select_features,
    split_data,
    train_xgboost,
    evaluate_xgboost,
    finalize_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=download_raw_data,
                inputs="params:xgboost.data_download",
                outputs="raw_data_from_source_xgb",
                name="xgboost_download_raw_data_node",
            ),
            node(
                func=preprocess_for_xgboost,
                inputs=["raw_data_from_source_xgb", "params:xgboost"],
                outputs="xgboost_processed_data",
                name="xgboost_preprocess_node",
            ),
            node(
                func=analyze_and_select_features,
                inputs=[
                    "xgboost_processed_data",
                    "params:xgboost.target_column",
                    "params:xgboost.feature_selection",
                    "params:xgboost.random_state",
                ],
                outputs=["selected_features_xgb", "feature_importances_xgb"],
                name="xgboost_feature_selection_node",
            ),
            node(
                func=split_data,
                inputs=[
                    "xgboost_processed_data",
                    "selected_features_xgb",
                    "params:xgboost.target_column",
                    "params:xgboost",
                ],
                outputs=["X_train_xgb", "X_test_xgb", "y_train_xgb", "y_test_xgb", "available_features_xgb"],
                name="xgboost_split_node",
            ),
            node(
                func=train_xgboost,
                inputs=[
                    "X_train_xgb",
                    "y_train_xgb",
                    "params:xgboost",
                    "params:xgboost.random_state",
                    "available_features_xgb",
                ],
                outputs=["xgboost_model", "train_run_id_xgb"],
                name="xgboost_train_node",
            ),
            node(
                func=evaluate_xgboost,
                inputs=[
                    "xgboost_model",
                    "X_train_xgb",
                    "X_test_xgb",
                    "y_train_xgb",
                    "y_test_xgb",
                    "feature_importances_xgb",
                    "train_run_id_xgb",
                ],
                outputs="evaluation_metrics_xgb",
                name="xgboost_evaluate_node",
            ),
            node(
                func=finalize_model,
                inputs=["xgboost_model", "available_features_xgb"],
                outputs="xgboost_final_artifacts",
                name="xgboost_finalize_node",
            ),
        ]
    )
