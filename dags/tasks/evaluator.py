"""
Untuk Validasi model mengggunakan fungsi evaluator(), argument yang berikan yaitu:
    model: model yang sudah dilatih [ model.train() ]
    dataloader_val: data test / data validation nya
    experiment_name: nama experiment mlflow untuk logging metrics

NOTE:
untuk data validation sementara ini masih menggunakan train_test_split yang dilakukan ML Engineer,
kedepannya akan didesain dan dibuatkan sendiri untuk Data Validation-nya, sehingga argument dari
evaluator() hanya model dan experiment_name
"""


import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, classification_report
import json
import mlflow
from tasks.load_dict import load_dict
from dotenv import load_dotenv

from airflow.decorators import task

def metrics_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    f1 = f1_score(labels_flat, preds_flat, average='weighted')
    acc = accuracy_score(labels_flat, preds_flat)
    precision = precision_score(labels_flat, preds_flat, average='weighted')
    recall = recall_score(labels_flat, preds_flat, average='weighted')
    
    print(classification_report(labels_flat, preds_flat))
    
    return f1, acc, precision, recall

def accuracy_per_class(preds, labels):
    label_dict_inverse = {v: k for k, v in load_dict().items()}
    
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')

def evaluator(model, dataloader_val, experiment_name):
    
    # label_dict = load_dict()

    mlflow.set_experiment(experiment_name=experiment_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_val:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
    
    f1_scores, acc_score, prec_score, recall_scores = metrics_func(predictions, true_vals)

    metrics = {
        "F1 Score": f1_scores,
        "Accuracy": acc_score,
        "Precision": prec_score,
        "Recall": recall_scores,
        "Validation loss": loss_val_avg
    }

    # mlflow.log_metrics(metrics=metrics)
    return metrics

@task(task_id='Evaluate-model')
def evaluate_model(flag):
    load_dotenv()
    if flag:
        print('Model Exist!')
    else :
        print("Model doesn't exist")