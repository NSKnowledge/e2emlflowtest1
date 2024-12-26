import json
from mlflow.tracking import MlflowClient
import mlflow
import dagshub

dagshub.init(repo_owner='NSKnowledge', repo_name='e2emlflowtest1', mlflow=True)
mlflow.set_experiment("Testrun1")
mlflow.set_tracking_uri("https://dagshub.com/NSKnowledge/e2emlflowtest1.mlflow")



reports_path = f"reports/run_info.json"

with open(reports_path,"r") as file:
    run_info = json.load(file)

run_id = run_info['run_id']
model_name = run_info['model_name']

client = MlflowClient()
model_uri = f"runs:/{run_id}/artifacts/{model_name}"

reg = mlflow.register_model(model_uri,model_name)
model_version = reg.version

new_Stage = "Staging"


client.transition_model_version_stage(
    name= model_name,
    version=model_version,
    stage=new_Stage,
    archive_existing_versions=True
)

print(f"model {model_name} version {model_version} transitioned to {new_Stage} stage.")