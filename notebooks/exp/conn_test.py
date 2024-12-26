import dagshub
dagshub.init(repo_owner='NSKnowledge', repo_name='e2emlflowtest1', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)