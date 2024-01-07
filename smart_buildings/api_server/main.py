from fastapi import FastAPI
from schema import PredictionRequest
from retrain.train import process_data, select_features, make_predictions
import polars as pl
import pickle

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
    print(type(predictions))
    return {"prediction": predictions.tolist()}
