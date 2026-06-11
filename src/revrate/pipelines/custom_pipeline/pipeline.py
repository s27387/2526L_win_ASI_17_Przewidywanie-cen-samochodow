from kedro.pipeline import Pipeline, node

from .nodes import (
    clean_data,
    impute_missing,
    engineer_features,
    analyze_and_select_features,
    split_data,
    preprocess_data,
    train_model,
    evaluate_model,
    finalize_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=clean_data,
                inputs=["raw_data", "params:custom.cleaning", "params:custom.target_column"],
                outputs="cleaned_data",
                name="clean_data_node",
            ),
            node(
                func=impute_missing,
                inputs=["cleaned_data", "params:custom.imputation"],
                outputs="imputed_data",
                name="impute_missing_node",
            ),
            node(
                func=engineer_features,
                inputs=["imputed_data", "params:custom.feature_engineering", "params:custom.target_column"],
                outputs="engineered_data",
                name="engineer_features_node",
            ),
            node(
                func=analyze_and_select_features,
                inputs=["engineered_data", "params:custom.target_column", "params:custom.feature_selection", "params:custom.random_state"],
                outputs=["selected_features", "feature_importances"],
                name="analyze_features_node",
            ),
            node(
                func=split_data,
                inputs=["engineered_data", "selected_features", "params:custom.target_column", "params:custom"],
                outputs=["X_train_raw", "X_test_raw", "y_train", "y_test"],
                name="split_data_node",
            ),
            node(
                func=preprocess_data,
                inputs=["X_train_raw", "X_test_raw", "params:custom.preprocessing"],
                outputs=["X_train", "X_test", "preprocessor"],
                name="preprocess_data_node",
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train", "params:custom.rf", "params:custom.random_state", "selected_features"],
                outputs=["trained_model", "train_run_id"],
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["trained_model", "X_train", "X_test", "y_train", "y_test", "feature_importances", "train_run_id"],
                outputs="evaluation_metrics",
                name="evaluate_model_node",
            ),
            node(
                func=finalize_model,
                inputs=["trained_model", "preprocessor", "selected_features"],
                outputs=["custom_model", "custom_preprocessor", "custom_top_features"],
                name="finalize_model_node",
            ),
        ]
    )
