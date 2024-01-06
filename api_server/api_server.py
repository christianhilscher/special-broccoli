from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
import numpy as np

# Dummy model training for demonstration
X, y = np.array([[1], [2], [3], [4], [5]]), np.array([1, 2, 3, 4, 5])
model = LinearRegression().fit(X, y)

# Define request schema
class PredictionRequest(BaseModel):
    data: list  # List of input features

app = FastAPI()

@app.post("/predict")
async def make_prediction(request: PredictionRequest):
    # Convert request data to numpy array for prediction
    input_data = np.array(request.data).reshape(-1, 1)
    prediction = model.predict(input_data)
    return {"prediction": prediction.tolist()}
