from pydantic import BaseModel
from typing import List

class SensorData(BaseModel):
    date: List[str]
    Temperature: List[float]
    Humidity: List[float]
    Light: List[float]
    CO2: List[float]
    HumidityRatio: List[float]
    Occupancy: List[int]

class PredictionRequest(BaseModel):
    data: SensorData