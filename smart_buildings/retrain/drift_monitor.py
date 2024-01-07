import requests
from train import read_data

if __name__ == "__main__":
    data_path = "~/Downloads/data/datatraining.txt"
    data = read_data(data_path)
    data_dict = {"data": data.sample(5).to_dict(as_series=False)}
    url = "http://localhost:5000/predict"

    response = requests.post(url, json=data_dict)
    print(response.text)