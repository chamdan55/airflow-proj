import datetime

import pendulum

from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator

with DAG(
    dag_id="run_python_via_CLI",
    schedule=None,
    catchup=False,
    tags=["CLI", "Pyton"],
) as dag:
    first = BashOperator(
        task_id='Initiate',
        bash_command='echo "Python will Running soon!"'
    )

    second = BashOperator(
        task_id ='Run-python',
        # bash_command='python "./tasks/taskBash.py"'
        # bash_command='cd .. && ls'
        bash_command='cd .. && python taskBash.py'
    )

    first >> second

if __name__ == "__main__":
    dag.test()