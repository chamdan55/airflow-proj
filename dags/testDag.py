from datetime import datetime
from airflow.decorators import dag

from tasks._testFunction import testFunction

@dag(
    schedule=None,
    start_date=datetime(2024, 6, 26),
    catchup=False,
    # owner="camdun",
    dag_id="testing_function",
    tags=["testing"],
)
def main():
    run = testFunction()

# if __name__ == "__main__":
main()