from datetime import datetime
from airflow import DAG
from airflow.decorators import dag, task

@task(task_id='printString')
def printString():
    print("Pipeline begin!")
    print("Task 1 being printed!")

@task(task_id='getNumber')
def getNumber():
    num1 = 8
    num2 = 15
    return num1, num2

@task(task_id='printType')
def printType(nums):
    print(nums[0], nums[1])
    print(type(nums))

@task(task_id='addNumber')
def addNumber(nums):
    print(f'Your Numbers are {nums[0]} & {nums[1]}')
    total = nums[0] + nums[1]
    print(f"The Result is : {total}")