from airflow.models import Variable
from airflow.providers.slack.notifications.slack_webhook import send_slack_webhook_notification
import pendulum
import os
from helpers.DagGenerator import DagGenerator
from google.cloud import storage
import tempfile

# Constants
PROJECT_NAME = "dp-floodLens"
PARTITION_DATE = "{{ dag_run.data_interval_end.astimezone(dag.timezone).strftime('%Y-%m-%d') }}"
TENANT = Variable.get("tenant")
ENVIRONMENT = Variable.get("environment")
AIRFLOW_ROOT = Variable.get("airflow_dag_root")
PROJECT_ROOT = f"{AIRFLOW_ROOT}/{PROJECT_NAME}"
CONFIG_FILE = f"{PROJECT_NAME}/conf/{TENANT}/{ENVIRONMENT}_config.json"
TIMEZONE = Variable.get("time_zone")
SLACK_WEBHOOK = "slack_webhook"

dag_failure_slack_webhook_notification = send_slack_webhook_notification(
    slack_webhook_conn_id=SLACK_WEBHOOK,
    text="*{{ dag.dag_id }}* DAG failed"
)
task_failure_slack_webhook_notification = send_slack_webhook_notification(
    slack_webhook_conn_id=SLACK_WEBHOOK,
    text="*{{ ti.task_id }}* failed in DAG {{ dag.dag_id }}"
)

default_args = {
    'owner': 'IWMI-E-Rewater',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
}

def download_gcs_file(gcs_path):
    """Download GCS file to local temporary file and return local path"""
    try:
        if not gcs_path.startswith("gs://"):
            # If it's already a local path, return as is
            return gcs_path

        # Parse GCS path
        path_parts = gcs_path.replace("gs://", "").split("/")
        bucket_name = path_parts[0]
        blob_name = "/".join(path_parts[1:])

        # Download from GCS
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        blob.download_to_filename(temp_file.name)

        print(f"Downloaded {gcs_path} to {temp_file.name}")
        return temp_file.name

    except Exception as e:
        print(f"Error downloading GCS file {gcs_path}: {str(e)}")
        raise e

gcs_config_path = f"gs://{AIRFLOW_ROOT}/{CONFIG_FILE}"
print(f"Downloading config from: {gcs_config_path}")
local_config_path = download_gcs_file(gcs_config_path)

try:
    dag_factory = DagGenerator(
        local_config_path,"dp-floodLens","dp-floodLens",default_args,catchup=False,start_date=pendulum.datetime(2026, 1, 28, tz=TIMEZONE)
        # schedule_interval='@monthly'
    )
    dag = dag_factory.generate_dag()

finally:
    if local_config_path and os.path.exists(local_config_path):
        os.remove(local_config_path)
        print(f"Cleaned up temporary file: {local_config_path}")