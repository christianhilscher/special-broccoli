import polars as pl
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import pathlib
import os



def read_data(path:str)->pl.DataFrame:
    data = pl.read_csv(path,has_header=False, skip_rows=1)
    header = pl.read_csv(path, n_rows=1, has_header=False, truncate_ragged_lines=True)
    data = data[:, 1:]
    data.columns = header.row(0)
    return data

def process_data(data:pl.DataFrame)->pl.DataFrame:
    data = data.with_columns(pl.col("date").str.to_datetime())
    data = data.with_columns(
        (pl.col("date").dt.hour().alias("hour_of_day")),
        (pl.col("date").dt.time().alias("time")),
        (pl.col("date").dt.weekday().alias("weekday")),
        (pl.col("date").dt.date().alias("date")),
    )
    return data

def select_features(data:pl.DataFrame, features:List[str])->pl.DataFrame:
    X = data.select(features)
    return X

def select_target(data:pl.DataFrame, target:str)->pl.Series:
    return data.select(target).to_series()

def train_model(X:pl.DataFrame, y:pl.Series, param_dict:Optional[Dict]=None)->lgb.LGBMClassifier:
    
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt',
        objective='binary',
        num_leaves=31,
        learning_rate=0.05,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=5
    )

    clf.fit(X, y)
    return clf

def make_predictions(X, trained_model):
    return trained_model.predict(X)

def evaluate_predictions(y_test:pl.Series, y_pred:pl.Series)-> None:
    print("Accuracy:", round(accuracy_score(y_test, y_pred),2))
    print("Precision:", round(precision_score(y_test, y_pred),2))
    print("Recall:", round(recall_score(y_test, y_pred),2))
    print("F1 Score:", round(f1_score(y_test, y_pred),2))

def save_trained_model(
        path:str, 
        trained_model:lgb.LGBMClassifier,
        id:str
        )->None:
    
    path = pathlib.Path(path)
    path.mkdir(exist_ok=True, parents=True)
    model_path = str(path / f"model-{id}.pkl")
    with open(model_path, 'wb') as file:
        pickle.dump(trained_model, file)

def run(data_path:str, model_path:str, id:str):
    data = read_data(path=data_path)
    data = process_data(data=data)
    X= select_features(data=data, features=["Light"])
    y = select_target(data=data, target="Occupancy")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    trained_model = train_model(X=X_train, y=y_train)
    predictions = make_predictions(X=X_test, trained_model=trained_model)
    evaluate_predictions(y_test=y_test, y_pred=predictions)
    save_trained_model(path=model_path, trained_model=trained_model, id=id)



if __name__ == "__main__":
    id = os.getenv("IDENTIFIER", "default-identifier")
    data_path = "~/Downloads/data/datatraining.txt"
    model_path = "/models"
    run(data_path, model_path, id)