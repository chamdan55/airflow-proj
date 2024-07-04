from airflow.decorators import task

@task(task_id='Print-params')
def printParams(params):
    try:
        for key, val in params.items():
            print(f"{key} is {val}")
        # print(type(params))
        # print(params)
    except Exception as e:
        print('THE ERROR IS :', e)
    