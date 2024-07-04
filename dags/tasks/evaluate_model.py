from airflow.decorators import task

@task(task_id='Evaluate-model')
def evaluate_model(flag):
    if flag:
        print('Model Exist!')
    else :
        print("Model doesn't exist")