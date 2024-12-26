import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
import yaml

def load_params(filepath:str)->int:
    try:
        with open(filepath,'r') as file:
            params = yaml.safe_load(file)
        return params["model_building"]["n_estimators"]
    except Exception as e:
        raise Exception(f"Error loading paramas {filepath}:{e}")

def load_data(filepath:str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error occured loading data{filepath}:{e}")
    
def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame,pd.Series]:
    try:
        X_train = data.drop(columns=['Potability'],axis=1)
        y_train = data['Potability']
        return X_train,y_train
    except Exception as e:
        raise Exception(f"Error occured preparing data:{e}")


def train_model(X_train:pd.DataFrame, y_train:pd.Series, n_estimator:int)-> RandomForestClassifier:
    try:
        clf = RandomForestClassifier(n_estimators=n_estimator)
        model = clf.fit(X_train,y_train)
        return model
    except Exception as e:
        raise Exception(f"Error training model:{e}")

def save_model (model: RandomForestClassifier, filepath:str) ->None:
    try:
        pickle.dump(model,open("models/model.pkl","wb"))
    except Exception as e:
        raise Exception(f"Error saving model:{e}")

def main():
    try:
        processed_data_path = r"./data/processed/train_processed.csv"
        params_path ="./params.yaml"
        model_name = "./models/model.pkl"
        n_estimator =load_params(params_path)
        processed_data = load_data(processed_data_path)
        train_processed_data, test_processed_data =prepare_data(processed_data)
        model = train_model(train_processed_data, test_processed_data, n_estimator)
        save_model(model,model_name)
 
    except Exception as e:
        raise Exception("An error occured: {e}")


if __name__=="__main__":
    main()