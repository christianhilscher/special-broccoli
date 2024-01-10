import json
from typing import List, Dict
import subprocess
import os

import polars as pl
import requests
from retrain.train import select_target
from retrain.utils import read_data, get_config
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def get_predictions(data: pl.DataFrame) -> List[int]:
    data_dict = {"data": data.to_dict(as_series=False)}
    api_server_url = os.getenv("API_SERVER_URL", "http://api-server:5000/predict")

    response = json.loads(requests.post(api_server_url, json=data_dict).text)
    return pl.Series(name="predictions", values=response["prediction"])


def get_model_evaluation_metrics(
    y_test: pl.Series, y_pred: pl.Series
) -> Dict[str, float]:
    metric_dict = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1-scrore": f1_score(y_test, y_pred),
    }
    print(metric_dict)
    return metric_dict


def get_data_evaluation_metrics():
    pass


def compare_metrics_against_thresholds(
    metrics: Dict[str, float], thresholds: Dict[str, float]
) -> bool:
    return any(
        metrics.get(metric, 0) < threshold for metric, threshold in thresholds.items()
    )


def trigger_retraining():
    try:
        # Run the Bash script
        subprocess.run(["/bin/bash", "/app/drift_monitor/raise_issue.sh"], check=True)
        print("GitHub issue created successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to create GitHub issue. Return code: {e.returncode}")
        if e.output:
            print(e.output.decode())


if __name__ == "__main__":
    config = get_config("/app/config/config.yaml")
    latest_data = read_data(config["test_data_path"])

    predictions = get_predictions(latest_data)
    real_values = select_target(latest_data, config["target"])

    model_evaluation_metrics = get_model_evaluation_metrics(real_values, predictions)
    model_drift = compare_metrics_against_thresholds(
        metrics=model_evaluation_metrics, thresholds=config["thresholds"]
    )
    model_drift = True

    if model_drift:
        trigger_retraining()
