# Databricks notebook source
# MAGIC %md
# MAGIC ## Monthly AutoML Retrain
# MAGIC 
# MAGIC <img src="https://github.com/RafiKurlansik/laughing-garbanzo/blob/main/step7.png?raw=true">

# COMMAND ----------

# DBTITLE 1,Load Features
from databricks.feature_store import FeatureStoreClient

# Set config for database name, file paths, and table names
feature_table = 'ibm_telco_churn.churn_features'

fs = FeatureStoreClient()

features = fs.read_table(feature_table)

# COMMAND ----------

# DBTITLE 1,Run AutoML
import databricks.automl
model = databricks.automl.classify(features, 
                                   target_col = "churn",
                                   data_dir= "dbfs:/tmp/",
                                   timeout_minutes=5) 

# COMMAND ----------

# DBTITLE 1,Register the Best Run
import mlflow
from mlflow.tracking.client import MlflowClient

client = MlflowClient()

run_id = model.best_trial.mlflow_run_id
model_name = "Telco_churn_model"
model_uri = f"runs:/{run_id}/model"

client.set_tag(run_id, key='db_table', value='sr_ibm_telco_churn.churn_features')
client.set_tag(run_id, key='demographic_vars', value='seniorCitizen,gender_Female')

model_details = mlflow.register_model(model_uri, model_name)

# COMMAND ----------

# DBTITLE 1,Add Descriptions
model_version_details = client.get_model_version(name=model_name, version=model_details.version)

client.update_registered_model(
  name=model_details.name,
  description="This model predicts whether a customer will churn using features from the ibm_telco_churn database.  It is used to update the Telco Churn Dashboard in DB SQL."
)

client.update_model_version(
  name=model_details.name,
  version=model_details.version,
  description="This model version was built using sklearn's LogisticRegression."
)

# COMMAND ----------

# DBTITLE 1,Request Transition to Staging
# Helper function
import mlflow
from mlflow.utils.rest_utils import http_request
import json

def client():
  return mlflow.tracking.client.MlflowClient()

host_creds = client()._tracking_client.store.get_host_creds()
host = host_creds.host
token = host_creds.token

def mlflow_call_endpoint(endpoint, method, body='{}'):
  if method == 'GET':
      response = http_request(
          host_creds=host_creds, endpoint="/api/2.0/mlflow/{}".format(endpoint), method=method, params=json.loads(body))
  else:
      response = http_request(
          host_creds=host_creds, endpoint="/api/2.0/mlflow/{}".format(endpoint), method=method, json=json.loads(body))
  return response.json()


# COMMAND ----------

# Transition request to staging
staging_request = {'name': model_name, 'version': model_details.version, 'stage': 'Staging', 'archive_existing_versions': 'true'}
mlflow_call_endpoint('transition-requests/create', 'POST', json.dumps(staging_request))

# COMMAND ----------

# Leave a comment for the ML engineer who will be reviewing the tests
comment = "This was the best model from AutoML, I think we can use it as a baseline."
comment_body = {'name': model_name, 'version': model_details.version, 'comment': comment}
mlflow_call_endpoint('comments/create', 'POST', json.dumps(comment_body))

# COMMAND ----------


