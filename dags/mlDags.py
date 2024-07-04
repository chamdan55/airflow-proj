from datetime import datetime
from airflow import DAG
from airflow.decorators import dag

# from tasks.data_ingestion import data_ingestion
# from tasks.pre_processing import pre_processing
# from tasks.training_preparation import training_preparation
# from tasks.hyperparameter_tuning import hyperparameter_tuning
# from dags.tasks._printParams import printParams

from pipelines.mlPipeline import ml_pipeline

@dag(
    schedule=None,
    start_date=datetime(2024, 6, 26),
    catchup=False,
    # owner="camdun",
    dag_id="ml_pipeline_test",
    tags=["testing"],
)
def main():
    run = ml_pipeline()

main()