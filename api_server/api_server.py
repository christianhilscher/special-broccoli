from fastapi import FastAPI
from pydantic import BaseModel

class PredictionRequest(BaseModel):
    data: list

app = FastAPI()

@app.post("/predict")
async def make_prediction(request: PredictionRequest):
    return {"prediction": request.data}
