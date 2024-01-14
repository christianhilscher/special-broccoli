import os
import pathlib
import pickle
from typing import List, Tuple

import numpy as np
import lightgbm as lgb
import polars as pl
from smart_buildings.logger import setup_logger
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from smart_buildings.utils import get_config, read_data

logger = setup_logger()
ID = os.getenv("IDENTIFIER")


def process_data(data: pl.DataFrame) -> pl.DataFrame:
    data = data.with_columns(pl.col("date").str.to_datetime())
    data = data.with_columns(
        (pl.col("date").dt.hour().alias("hour_of_day")),
        (pl.col("date").dt.time().alias("time")),
        (pl.col("date").dt.weekday().alias("weekday")),
        (pl.col("date").dt.date().alias("date")),
    )
    return data


def temporal_split(
    data: pl.DataFrame, validation_days: int
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    data = data.with_columns(
        (
            pl.col("date") >= (pl.col("date").max() - pl.duration(days=validation_days))
        ).alias("validation")
    )

    _log_split_fraction(data, validation_days)

    return data.filter(~pl.col("validation")), data.filter(pl.col("validation"))


def _log_split_fraction(data, validation_days):
    logger.info(
        "{} validation days yield a {:.2f} test fraction".format(
            validation_days, data["validation"].mean()
        )
    )


def select_features(data: pl.DataFrame, features: List[str]) -> pl.DataFrame:
    return data.select(features)


def select_target(data: pl.DataFrame, target: str) -> pl.Series:
    return data.select(target).to_series()


def train_model(X: pl.DataFrame, y: pl.Series) -> lgb.LGBMClassifier:
    clf = lgb.LGBMClassifier(
        boosting_type="gbdt",
        objective="binary",
        max_depth=np.random.randint(3, 9),
        bagging_fraction=0.7,
    )

    clf.fit(X, y)
    return clf


def make_predictions(X: pl.DataFrame, trained_model: lgb.LGBMClassifier):
    return trained_model.predict(X)


def evaluate_predictions(y_test: pl.Series, y_pred: pl.Series) -> None:
    logger.info(
        "Accuracy: %s",
        round(accuracy_score(y_test, y_pred), 2),
        extra={"accuracy": round(accuracy_score(y_test, y_pred), 2)},
    )
    logger.info(
        "Precision: %s",
        round(precision_score(y_test, y_pred), 2),
        extra={"precision": round(precision_score(y_test, y_pred), 2)},
    )
    logger.info(
        "Recall: %s",
        round(recall_score(y_test, y_pred), 2),
        extra={"recall": round(recall_score(y_test, y_pred), 2)},
    )
    logger.info(
        "F1-Score: %s",
        round(f1_score(y_test, y_pred), 2),
        extra={"f1-score": round(f1_score(y_test, y_pred), 2)},
    )


def save_trained_model(path: str, trained_model: lgb.LGBMClassifier) -> None:
    path = pathlib.Path(path)
    path.mkdir(exist_ok=True, parents=True)
    model_path = str(path / f"model-{ID}.pkl")
    with open(model_path, "wb") as file:
        pickle.dump(trained_model, file)


def run_training(
    train_data: pl.DataFrame,
    test_data: pl.DataFrame,
    features: List[str],
    target: str,
) -> lgb.LGBMClassifier:
    trained_model = train_model(
        X=select_features(train_data, features), y=select_target(train_data, target)
    )
    predictions = make_predictions(
        X=select_features(test_data, features), trained_model=trained_model
    )
    evaluate_predictions(y_test=select_target(test_data, target), y_pred=predictions)
    return trained_model


def fetch_data(data_path: str) -> pl.DataFrame:
    data = read_data(path=data_path)
    data = process_data(data=data)
    return data


if __name__ == "__main__":
    config = get_config("/app/config/config.yaml")

    train_data = fetch_data(data_path=config["train_data_path"])
    test_data = fetch_data(data_path=config["test_data_path"])

    trained_model = run_training(
        train_data=train_data,
        test_data=test_data,
        features=config["training"]["features"],
        target=config["training"]["target"],
    )

    save_trained_model(config["model_path"], trained_model)
