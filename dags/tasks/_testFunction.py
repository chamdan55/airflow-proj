from airflow.decorators import task
import json
def add(x,y):
    return x+y

@task(task_id='test-Function')
def testFunction():
    with open('./data/label_dict.json') as json_file:
        label_dict = json.load(json_file)
    print(label_dict)