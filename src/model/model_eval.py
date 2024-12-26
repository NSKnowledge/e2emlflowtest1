import numpy as np
import pandas as pd
import pickle
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score, confusion_matrix
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from mlflow.models import infer_signature
import dagshub

dagshub.init(repo_owner='NSKnowledge', repo_name='e2emlflowtest1', mlflow=True)
mlflow.set_experiment("Testrun1")
mlflow.set_tracking_uri("https://dagshub.com/NSKnowledge/e2emlflowtest1.mlflow")

def load_data(filepath:str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error Loading data from {filepath}:{e}")

# test_data = pd.read_csv("./data/processed/test_processed.csv")

def prepare_data(data:pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    try:
        X_test = data.drop(columns=['Potability'],axis=1)
        y_test = data['Potability']
        return X_test,y_test
    except Exception as e:
        raise Exception(f"Error preparing data:{e}")
    
def load_model(filepath:str):
    try:
        with open(filepath,"rb") as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        raise Exception(f"Error loading model:{e}")
    

def save_metrics(metrics:dict, filepath:str) ->None:
    try:
        with open(filepath,"w") as file:
            json.dump(metrics,file,indent=4)
    except Exception as e:
        raise Exception(f"Error saving metrics:{e}")
    
def eval_model(model,X_test:pd.DataFrame, y_test:pd.Series,model_name:str) -> dict:
    try:
        params = yaml.safe_load(open("params.yaml", "r"))
        test_size = params["data_collection"]["test_size"]
        n_estimators = params["model_building"]["n_estimators"]
        
        y_pred = model.predict(X_test)

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        mlflow.log_param("Test_size",test_size)
        mlflow.log_param("n_estimators",n_estimators) 

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix for {model_name}")
        cm_path = f"confusion_matrix_{model_name.replace(' ', '_')}.png"
        plt.savefig(cm_path)
        
        mlflow.log_artifact(cm_path)
        
        metrics_dict = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        return metrics_dict
    except Exception as e:
        raise Exception(f"Error evaluating model: {e}")

def main():
    try:
        train_data_path = "./data/processed/train_processed.csv"
        test_data_path = "./data/processed/test_processed.csv"
        model_path = "models/model.pkl"
        metrics_path = "reports/metrics.json"
        model_name = "Best Model"
        train_data = load_data(train_data_path)
        test_data = load_data(test_data_path)
        X_test, y_test = prepare_data(test_data)
        model = load_model(model_path)

        # Start MLflow run
        with mlflow.start_run() as run:
            metrics = eval_model(model, X_test, y_test, model_name)
            save_metrics(metrics, metrics_path)

            # Log artifacts
            mlflow.log_artifact(model_path)
            mlflow.log_artifact(metrics_path)

            training_data = mlflow.data.from_pandas(train_data)
            testing_data = mlflow.data.from_pandas(test_data)
            mlflow.log_input(training_data,"training_data")
            mlflow.log_input(testing_data,"test _data")
            
            # Log the source code file
            mlflow.log_artifact(__file__)

            signature = infer_signature(X_test,model.predict(X_test))

            mlflow.sklearn.log_model(model,"Best Model",signature=signature)

            #Save run ID and model info to JSON File
            run_info = {'run_id': run.info.run_id, 'model_name': "Best Model"}
            reports_path = "reports/run_info.json"
            with open(reports_path, 'w') as file:
                json.dump(run_info, file, indent=4)
        
    except Exception as e:
        raise Exception(f"An error occured :{e}")


if __name__=="__main__":
    main()

