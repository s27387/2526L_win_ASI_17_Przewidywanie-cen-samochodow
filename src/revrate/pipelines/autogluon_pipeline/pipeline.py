from kedro.pipeline import Pipeline, node

from .nodes import (
    preprocess_for_autogluon,
    split_data,
    train_autogluon,
    evaluate_autogluon,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=preprocess_for_autogluon,
                inputs=["raw_data", "params:autogluon.target_column"],
                outputs="autogluon_processed_data",
                name="autogluon_preprocess_node",
            ),
            node(
                func=split_data,
                inputs=["autogluon_processed_data", "params:autogluon"],
                outputs=["X_train_ag", "X_test_ag", "y_train_ag", "y_test_ag"],
                name="autogluon_split_node",
            ),
            node(
                func=train_autogluon,
                inputs=["X_train_ag", "y_train_ag", "params:autogluon"],
                outputs=["autogluon_predictor", "train_run_id_ag"],
                name="autogluon_train_node",
            ),
            node(
                func=evaluate_autogluon,
                inputs=["autogluon_predictor", "X_test_ag", "y_test_ag", "train_run_id_ag"],
                outputs="autogluon_metrics",
                name="autogluon_evaluate_node",
            ),
        ]
    )
