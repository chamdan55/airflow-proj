from transformers import BertForSequenceClassification, BertTokenizer
import torch
import numpy as np
import json
import os
import mlflow

from airflow.decorators import task
from tasks.load_dict import load_dict

save_directory = './model_artifacts'
reg_model_name = 'Finetuned_[model]'

@task(task_id='Upload-model')
def upload_model(dicts):
    try:
        save_directory = './model_artifacts'
        exp_name = 'test_airflow'
        reg_model_name = 'Finetuned_[model]'
        print('Uploading model to MLFlow ... ... ...')
        # 1. Find the Maximum Accuracy
        max_accuracy = max(entry['Accuracy'] for entry in dicts.values())

        # 2. Find the Key(s) with the Maximum Accuracy
        best_indices = [key for key, entry in dicts.items() if entry['Accuracy'] == max_accuracy][0]

        # 3. Get the path of the first best index
        best_path = dicts[best_indices]['path']

        # Define and log the custom wrapper
        class BERTModelWrapper(mlflow.pyfunc.PythonModel):
            def load_context(self, context):
                
                with open(os.path.join(context.artifacts["model_artifacts"], "label_dict_inv.json"), "r") as f:
                    self.label_dict_inv = json.load(f)
                
                model_file_path = os.path.join(context.artifacts["model_artifacts/state_dict"], f"finetuned_BERT_epoch_{best_indices}.model")
                self.model = BertForSequenceClassification.from_pretrained("indolem/indobertweet-base-uncased", num_labels=len(self.label_dict_inv))
                self.model.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))
                self.tokenizer = BertTokenizer.from_pretrained(context.artifacts["model_artifacts/tokenizer"])
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.model.to(self.device)

            def predict(self, context, model_input):
                texts = model_input["texts"].tolist()
                inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
                predicted_labels = np.argmax(probabilities, axis=1)
                predicted_labels = [self.label_dict_inv[str(label)] for label in predicted_labels]
                return predicted_labels
        mlflow.set_experiment(exp_name)
        with mlflow.start_run():
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=BERTModelWrapper(),
                artifacts={
                    "model_artifacts": save_directory,
                    "tokenizer": os.path.join(save_directory, "tokenizer")
                },
                registered_model_name=reg_model_name
            )
        print('Upload Succeed!')
        return 1
    except Exception as e:
        print('Upload Failed!')
        print('THE ERROR IS :', e)
        return 0
