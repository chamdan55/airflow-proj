from tasks.data_ingestion import data_ingestion
from tasks.pre_processing import pre_processing
from tasks.training_preparation import training_preparation
from tasks.hyperparameter_tuning import hyperparameter_tuning
from tasks.finetuning_model import finetuning_model
from tasks.upload_model import upload_model
from tasks.evaluator import evaluate_model

def ml_pipeline():
    data = data_ingestion()
    data = pre_processing(data=data)
    data = training_preparation(data)
    lr = hyperparameter_tuning()
    state_dict = finetuning_model(data=data, lr=lr)
    flag = upload_model(state_dict)
    evaluate_model(flag)
    # promote_model()