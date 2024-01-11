import os
import pathlib
import pickle
from typing import Dict, List, Optional, Tuple
from retrain.utils import get_config, read_data


import lightgbm as lgb
import polars as pl
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from logger import setup_logger

logger = setup_logger()


def process_data(data: pl.DataFrame) -> pl.DataFrame:
    data = data.with_columns(pl.col("date").str.to_datetime())
    data = data.with_columns(
        (pl.col("date").dt.hour().alias("hour_of_day")),
        (pl.col("date").dt.time().alias("time")),
        (pl.col("date").dt.weekday().alias("weekday")),
        (pl.col("date").dt.date().alias("date")),
    )
    return data


def train_test_split(
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
    X = data.select(features)
    return X


def select_target(data: pl.DataFrame, target: str) -> pl.Series:
    return data.select(target).to_series()


def train_model(
    X: pl.DataFrame, y: pl.Series, param_dict: Optional[Dict] = None
) -> lgb.LGBMClassifier:
    clf = lgb.LGBMClassifier(
        boosting_type="gbdt",
        objective="binary",
        num_leaves=31,
        learning_rate=0.05,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=5,
    )

    clf.fit(X, y)
    return clf


def make_predictions(X, trained_model):
    return trained_model.predict(X)


def evaluate_predictions(y_test: pl.Series, y_pred: pl.Series) -> None:
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 2))
    print("Precision:", round(precision_score(y_test, y_pred), 2))
    print("Recall:", round(recall_score(y_test, y_pred), 2))
    print("F1 Score:", round(f1_score(y_test, y_pred), 2))


def save_trained_model(path: str, trained_model: lgb.LGBMClassifier, id: str) -> None:
    path = pathlib.Path(path)
    path.mkdir(exist_ok=True, parents=True)
    model_path = str(path / f"model-{id}.pkl")
    with open(model_path, "wb") as file:
        pickle.dump(trained_model, file)


def run_training(
    data_path: str, features: List[str], target: str, validation_days: int
):
    data = read_data(path=data_path)
    data = process_data(data=data)
    train, test = train_test_split(data=data, validation_days=validation_days)
    trained_model = train_model(
        X=select_features(train, features), y=select_target(train, target)
    )
    predictions = make_predictions(
        X=select_features(test, features), trained_model=trained_model
    )
    evaluate_predictions(y_test=select_target(test, target), y_pred=predictions)


if __name__ == "__main__":
    id = os.getenv("IDENTIFIER", "")
    config = get_config("/app/config/config.yaml")

    trained_model = run_training(
        data_path=config["train_data_path"],
        features=config["features"],
        target=config["target"],
        validation_days=config["validation_days"],
    )
    save_trained_model(config["model_path"], trained_model, id)
