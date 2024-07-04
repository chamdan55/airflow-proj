# WHAT LOGGED TO MLFLOW IN THIS FILE:
# label_dict.json : ./model_artifacts/label_dict.json
# label_dict_inv.json : ./model_artifacts/label_dict_inv.json
# tokenizer : ./model_artifacts/tokenizer/

from datetime import datetime
import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
import mlflow
from dotenv import load_dotenv

from airflow.decorators import task
import os

save_directory = './model_artifacts'

@task(task_id='Data-preparation')
def training_preparation(data):
    try:
        load_dotenv()
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        print("Executing training preparation")
        # now = datetime.now()
        # date_formated = now.strftime("%d-%m-%Y")

        # # read .csv data
        # data = pd.read_csv(f"data/pre-processed_data/pre-processed_data_{date_formated}.csv")
        # # data = pd.read_csv("data/test_data/cleaned_final.csv")
        
        # # create label dictionary
        # # possible_labels = data.category.unique()
        # # label_dict = {}
        # # for index, possible_label in enumerate(possible_labels):
        # #     label_dict[possible_label] = index
        # # print(label_dict)

        print(data["category"].unique())
        category_map = {
            "Uang Keluar": 1,
            "Tabungan & Investasi": 2,
            "Pinjaman": 3,
            "Tagihan": 4,
            "Hadiah & Amal": 5,
            "Transportasi": 6,
            "Belanja": 7,
            "Top Up": 8,
            "Hiburan": 9,
            "Makanan & Minuman": 10,
            "Biaya & Lainnya": 11,
            "Hobi & Gaya Hidup": 12,
            "Perawatan Diri": 13,
            "Kesehatan": 14,
            "Pendidikan": 15,
            "Uang Masuk": 16,
            "Gaji": 17,
            "Pencairan Investasi": 18,
            "Bunga": 19,
            "Refund": 20,
            "Pencairan Pinjaman": 21,
            "Cashback": 22
        }

        # Create the inverse label dictionary
        label_dict_inv = {v: k for k, v in category_map.items()}

        mlflow.set_experiment('BertModel')
        with mlflow.start_run():
            mlflow.log_dict(category_map, "./model_artifacts/label_dict.json")
            mlflow.log_dict(label_dict_inv, "./model_artifacts/label_dict_inv.json")

        # create label column
        print('Mapping category ... ... ...')
        data["label"] = data.category.map(category_map)

        print(data["category"].value_counts())

        val = data.category.value_counts()
        filtered_val = val[val>=2]
        mask = data['category'].isin(filtered_val.index)
        data = data[mask]
        

        # # splitting data
        # print('Splitting data into train and validation set ... ... ...')
        # X_train, X_val, y_train, y_val = train_test_split(data.index.values, 
        #                                                 data.label.values, 
        #                                                 test_size=0.10, 
        #                                                 random_state=42, 
        #                                                 stratify=data.label.values)
        # data['data_type'] = ['not_set']*data.shape[0]

        # data.loc[X_train, 'data_type'] = 'train'
        # data.loc[X_val, 'data_type'] = 'val'

        # # data splitting result
        # print(data.groupby(['category', 'label', 'data_type']).count())

        # # tokenize dataset
        # print('Preparing tokenizer ... ... ...')
        # tokenizer = BertTokenizer.from_pretrained('indolem/indobertweet-base-uncased', do_lower_case=True)
        
        # # Log Tokenizer to MLFlow
        # print('Logging Tokenizer to MLFlow ... ... ...')
        # path_dir = os.path.join(save_directory, "tokenizer")
        # tokenizer.save_pretrained(path_dir)
        # with mlflow.start_run():
        #     mlflow.log_artifacts(path_dir, path_dir)

        # encoded_data_train = tokenizer.batch_encode_plus(
        #     data[data.data_type=='train'].merchantname.values, 
        #     add_special_tokens=True, 
        #     return_attention_mask=True, 
        #     pad_to_max_length=True, 
        #     max_length=256, 
        #     return_tensors='pt'
        # )

        # encoded_data_val = tokenizer.batch_encode_plus(
        #     data[data.data_type=='val'].merchantname.values, 
        #     add_special_tokens=True, 
        #     return_attention_mask=True, 
        #     pad_to_max_length=True, 
        #     max_length=256, 
        #     return_tensors='pt'
        # )

        # print('Splitting: input_ids, attetion_mask, labels ... ... ...')
        # input_ids_train = encoded_data_train['input_ids']
        # attention_masks_train = encoded_data_train['attention_mask']
        # labels_train = torch.tensor(data[data.data_type=='train'].label.values)

        # input_ids_val = encoded_data_val['input_ids']
        # attention_masks_val = encoded_data_val['attention_mask']
        # labels_val = torch.tensor(data[data.data_type=='val'].label.values)

        # print('Making TensorDataset for Train and Validation set ... ... ...')
        # dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
        # dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)
        # print("Train data: ", dataset_train)
        # print("Validation data: ", dataset_val)

        # now = datetime.now()
        # date_formated = now.strftime("%d-%m-%Y")
        # torch.save(dataset_train, f"data/tensor_dataset/dataset_train_{date_formated}.pt")
        # torch.save(dataset_val, f"data/tensor_dataset/dataset_val_{date_formated}.pt")

        # # print("Batching the data(s) ... ... ...")
        # # batch_size = 64
        # # dataloader_train = DataLoader(dataset_train,
        # #                             sampler=RandomSampler(dataset_train),
        # #                             batch_size=batch_size)

        # # dataloader_validation = DataLoader(dataset_val,
        # #                                 sampler=SequentialSampler(dataset_val),
        # #                                 batch_size=batch_size)
        
        # data_dict = {
        #     'train': dataset_train, #dataloader_train,
        #     'validation': dataset_val #dataloader_validation
        # }
        # print("Training preparation SUCCEED!")
        # print("Returning TensorDataset for Train and Validation in dictionary.")
        return data
            
    except Exception as e:
        print("Training preparation FAILED!")
        print('THE ERROR IS :', e)
