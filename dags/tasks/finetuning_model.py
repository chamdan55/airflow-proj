# WHAT LOGGED TO MLFLOW IN THIS FILE:
# metrics : train_loss_avg, f1_scores, acc_score, prec_score, recall_scores, loss_val_avg
#           epoch_time, gpu_usage, memory_usage, cpu_usage
# state_dict (per epoch) : .\model_artifacts\state_dict\finetuned_BERT_epoch_{epoch+1}.model'

from transformers import AdamW, get_linear_schedule_with_warmup, BertForSequenceClassification, BertTokenizer
import os
import torch
import random
import mlflow
import time
import psutil
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from datetime import datetime, timedelta

from airflow.decorators import task

from tasks.load_dict import load_dict
from tasks.evaluator import evaluator

from dotenv import load_dotenv

save_directory = './model_artifacts'
exp_name = 'BertModel'
reg_model_name = 'Finetuned_[model]'

@task(task_id='Finetuning-Model',
      execution_timeout=timedelta(days=7))
def finetuning_model(data, lr):
    try:
        save_directory = './model_artifacts'
        exp_name = 'test_airflow'
        reg_model_name = 'Finetuned_[model]'
        load_dotenv()
        if lr: 
            print('Success passing the parameters!')
            print('Your LR is: ', lr)
            
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        print(data)
        seed_val = 17
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        epochs = 1
        label_dict = load_dict()

        # splitting data
        print('Splitting data into train and validation set ... ... ...')
        X_train, X_val, y_train, y_val = train_test_split(data.index.values, 
                                                        data.label.values, 
                                                        test_size=0.10, 
                                                        random_state=42, 
                                                        stratify=data.label.values)
        data['data_type'] = ['not_set']*data.shape[0]

        data.loc[X_train, 'data_type'] = 'train'
        data.loc[X_val, 'data_type'] = 'val'

        # data splitting result
        print(data.groupby(['category', 'label', 'data_type']).count())

        # tokenize dataset
        print('Preparing tokenizer ... ... ...')
        tokenizer = BertTokenizer.from_pretrained('indolem/indobertweet-base-uncased', do_lower_case=True)
        
        # Log Tokenizer to MLFlow
        print('Logging Tokenizer to MLFlow ... ... ...')
        path_dir = os.path.join(save_directory, "tokenizer")
        tokenizer.save_pretrained(path_dir)
        with mlflow.start_run():
            mlflow.log_artifacts(path_dir, path_dir)

        encoded_data_train = tokenizer.batch_encode_plus(
            data[data.data_type=='train'].merchantname.values, 
            add_special_tokens=True, 
            return_attention_mask=True, 
            pad_to_max_length=True, 
            max_length=256, 
            return_tensors='pt'
        )

        encoded_data_val = tokenizer.batch_encode_plus(
            data[data.data_type=='val'].merchantname.values, 
            add_special_tokens=True, 
            return_attention_mask=True, 
            pad_to_max_length=True, 
            max_length=256, 
            return_tensors='pt'
        )

        print('Splitting: input_ids, attetion_mask, labels ... ... ...')
        input_ids_train = encoded_data_train['input_ids']
        attention_masks_train = encoded_data_train['attention_mask']
        labels_train = torch.tensor(data[data.data_type=='train'].label.values)

        input_ids_val = encoded_data_val['input_ids']
        attention_masks_val = encoded_data_val['attention_mask']
        labels_val = torch.tensor(data[data.data_type=='val'].label.values)

        print('Making TensorDataset for Train and Validation set ... ... ...')
        dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
        dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)
        print("Train data: ", dataset_train)
        print("Validation data: ", dataset_val)

        now = datetime.now()
        date_formated = now.strftime("%d-%m-%Y")
        torch.save(dataset_train, f"data/tensor_dataset/dataset_train_{date_formated}.pt")
        torch.save(dataset_val, f"data/tensor_dataset/dataset_val_{date_formated}.pt")

        print("Batching the data(s) ... ... ...")
        batch_size = 64
        dataloader_train = DataLoader(dataset_train,
                                    sampler=RandomSampler(dataset_train),
                                    batch_size=batch_size)

        dataloader_validation = DataLoader(dataset_val,
                                        sampler=SequentialSampler(dataset_val),
                                        batch_size=batch_size)
        
        print('Success getting Data!')

        print('Loading model ... ... ...')
        model = BertForSequenceClassification.from_pretrained("indolem/indobertweet-base-uncased",
                                                            num_labels=len(label_dict),
                                                            output_attentions=False,
                                                            output_hidden_states=False)

        optimizer = AdamW(model.parameters(),
                        lr=lr,
                        eps=1e-8)

        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=len(dataloader_train)*epochs)

        # Make sure to define device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        mlflow.set_experiment(exp_name)
        # mlflow.enable_system_metrics_logging()
        # mlflow.autolog()
        print("Training model is starting ... ... ...")
        with mlflow.start_run():
            acc_dict = {}
            for epoch in range(0, epochs):
                print('Starting Epoch: ', epoch+1)

                epoch_start_time = time.time()

                model.train()
                loss_train_total = 0

                for batch in dataloader_train:
                    model.zero_grad()

                    batch = tuple(b.to(device) for b in batch)
                    inputs = {'input_ids': batch[0],
                            'attention_mask': batch[1],
                            'labels': batch[2],
                            }

                    outputs = model(**inputs)
                    loss = outputs[0]
                    loss_train_total += loss.item()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()

                train_loss_avg = loss_train_total / len(dataloader_train)
                mlflow.log_metric("train_loss_avg", train_loss_avg, step=epoch)

                metrics = evaluator(model, dataloader_validation, exp_name)
                metrics["epoch_time"] = time.time() - epoch_start_time
                
                # metrics that recorded are:
                # F1 Score, Accuracy, Precision, Recall
                # diatas bisa ditambah metrix lagi
                mlflow.log_metrics(metrics=metrics, step=epoch)

                # Log system metrics
                mlflow.log_metrics({
                    "cpu_usage": psutil.cpu_percent(),
                    "memory_usage": psutil.virtual_memory().percent,
                    "gpu_usage": torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
                }, step=epoch)

                # Save the model checkpoint to MLflow
                path = f'{save_directory}/state_dict/finetuned_BERT_epoch_{epoch+1}.model'
                torch.save(model.state_dict(), path)
                mlflow.log_artifact(path, path)
                temp_dict = {
                    'path': path,
                    'Accuracy': metrics['Accuracy']
                }
                acc_dict[epoch+1] = temp_dict
                # mlflow.pytorch.log_model(model, "model", registered_model_name=reg_model_name)
        print('Training model is DONE!')
        return acc_dict
    
    except Exception as e:
        print("Finetune Model FAILED!")
        print('THE ERROR IS :', e)