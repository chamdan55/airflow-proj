from datetime import datetime
from airflow.decorators import dag
from tasks._additionT import *
from pipelines.additionP import additionPipeline

# @dag(
#     schedule=None,
#     start_date=datetime(2024, 6, 26),
#     catchup=False,
#     # owner="camdun",
#     dag_id="addition_pipeline",
#     tags=["testing"],
# )
# def additionPipeline():
#     printString()
#     nums = getNumber()
#     addNumber(nums)

@dag(
    schedule=None,
    start_date=datetime(2024, 6, 26),
    catchup=False,
    # owner="camdun",
    dag_id="addition_pipeline",
    tags=["testing"],
)
def main():
    run = additionPipeline()

# # if __name__=="__main__":
#     # run = additionPipeline()

# # run = additionPipeline()

# if __name__ == "__main__":
main()