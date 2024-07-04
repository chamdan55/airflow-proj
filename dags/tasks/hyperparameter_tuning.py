from airflow.decorators import task
import random

@task(task_id="hyperparameter_tuning")
def hyperparameter_tuning():
    try:
        print("Executing hyperparameter tuning")
        # optimum_params = {
        #     'param1': 0.1,
        #     'param2': 0.5,
        #     'param3': 10
        # }
        lr = [1e-3, 1e-4, 1e-5]
        optimum_params = random.choice(lr)
        print('Success! Returning: ', optimum_params)
        return optimum_params
    except Exception as e:
        print('Failed getting parameters')
        print('THE ERROR IS :', e)