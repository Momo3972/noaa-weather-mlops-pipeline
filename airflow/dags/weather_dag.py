from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2026, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'NOAA_Weather_Auto_Training',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False
) as dag:

    # 1. Tâche d'entraînement (Unique tâche active)
    run_training = BashOperator(
        task_id='execute_training_script',
        bash_command='python3 /opt/airflow/src/train.py'
    )