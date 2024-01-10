import pickle

import polars as pl
from fastapi import FastAPI
from retrain.train import make_predictions, process_data, select_features
from schema import PredictionRequest

app = FastAPI()


def get_model():
    with open("/app/models/model.pkl", "rb") as file:
        model = pickle.load(file)
    return model


@app.post("/predict")
async def make_prediction(request: PredictionRequest):
    print("Received request data:", request.data)
    data = pl.DataFrame._from_dict(request.data.model_dump())
    data = process_data(data=data)
    X = select_features(data, ["Light"])
    trained_model = get_model()
    predictions = make_predictions(X=X, trained_model=trained_model)
    print(predictions)
    return {"prediction": predictions.tolist()}
