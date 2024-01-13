import requests
from utils import read_data

if __name__ == "__main__":
    data_path = "~/Downloads/data/datatest.txt"
    data = read_data(data_path)
    data_dict = {"data": data.sample(5).to_dict(as_series=False)}
    url = "http://51.20.8.35:80/predict"
    # url = "http://localhost:5000/predict"

    response = requests.post(url, json=data_dict)
    print(response.text)
    print("real", data_dict["data"]["Occupancy"])
