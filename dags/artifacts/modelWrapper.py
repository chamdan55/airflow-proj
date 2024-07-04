from transformers import BertForSequenceClassification, BertTokenizer
import os
import mlflow
import torch
import json
import numpy as np

# Define and log the custom wrapper
class BERTModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        model_file_path = os.path.join(context.artifacts["model_artifacts"], "finetuned_BERT_epoch_1.model")
        self.model = BertForSequenceClassification.from_pretrained("indolem/indobertweet-base-uncased", num_labels=len(label_dict))
        self.model.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))
        self.tokenizer = BertTokenizer.from_pretrained(context.artifacts["tokenizer_dir"])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        with open(os.path.join(context.artifacts["model_artifacts"], "label_dict_inv.json"), "r") as f:
            self.label_dict_inv = json.load(f)

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