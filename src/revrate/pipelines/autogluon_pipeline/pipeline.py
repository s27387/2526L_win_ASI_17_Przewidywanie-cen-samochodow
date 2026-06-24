from kedro.pipeline import Pipeline, node

from ..custom_pipeline.nodes import download_raw_data

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
                func=download_raw_data,
                inputs="params:custom.data_download",
                outputs="raw_data_from_source_ag",
                name="autogluon_download_raw_data_node",
            ),
            node(
                func=preprocess_for_autogluon,
                inputs=["raw_data_from_source_ag", "params:autogluon"],
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
