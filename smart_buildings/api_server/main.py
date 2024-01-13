import pickle
from typing import Dict, List

import lightgbm as lgb
import polars as pl
import yaml
from fastapi import FastAPI
from retrain.train import make_predictions, process_data, select_features
from schema import PredictionRequest

app = FastAPI()


def get_model(path: str) -> lgb.LGBMClassifier:
    with open(path, "rb") as file:
        model = pickle.load(file)
    return model


def get_config(path: str) -> Dict[str, str]:
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config


def prepare_data(data: dict, features: List[str]) -> pl.DataFrame:
    data = pl.DataFrame._from_dict(data)
    data = process_data(data=data)
    return select_features(data=data, features=features)


@app.post("/predict")
async def make_prediction(request: PredictionRequest):
    config = get_config(path="/app/config/config.yaml")
    data = prepare_data(
        data=request.data.model_dump(), features=config["training"]["features"]
    )

    trained_model = get_model(path="/app/models/model.pkl")
    predictions = make_predictions(X=data, trained_model=trained_model)

    return {"prediction": predictions.tolist()}
