from datetime import datetime
from airflow import DAG
from airflow.decorators import dag, task
from airflow.operators.python_operator import PythonOperator

from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker
from tasks.models import Merchant as ModelMerchant
from tasks.models import Category as ModelCategory
from tasks.models import MerchantGarage as ModelMerchantGarage
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import os

# default_args = {
#     "owner": "adhitya primandhika",
#     "start_date": datetime(2024, 6, 12)
# }

# dag = DAG("ml-pipeline", default_args=default_args, schedule_interval=None)
@dag(
    schedule=None,
    start_date=datetime(2021, 1, 1),
    catchup=False,
    # owner="adhitya primandhika",
    dag_id="ml_pipeline",
    tags=["ml-pipeline"],
)
def ml_pipeline():
    @task(task_id='data_ingestion')
    def data_ingestion():
        print("Executing data ingestion")

        # load .env values
        load_dotenv()

        # load configuration for connecting to Postgresql
        engine = create_engine(os.getenv("DB_URL"))
        Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        session = Session()

        # query all data in merchant_garage
        merchants = session.query(ModelMerchantGarage).all()
        names = []
        category_ids = []
        for merchant in merchants:
            if merchant.name != "":
                names.append(merchant.name)
            else: 
                names.append(merchant.sub_name)
            category_ids.append(merchant.category_id)
        
        # create dataframe 
        df = pd.DataFrame({"merchantname": names, "category": category_ids})
        
        category_map = {
            1: "Uang Keluar",
            2: "Tabungan & Investasi",
            3: "Pinjaman",
            4: "Tagihan",
            5: "Hadiah & Amal",
            6: "Transportasi",
            7: "Belanja",
            8: "Top Up",
            9: "Hiburan",
            10: "Makanan & Minuman",
            11: "Biaya & Lainnya",
            12: "Hobi & Gaya Hidup",
            13: "Perawatan Diri",
            14: "Kesehatan",
            15: "Pendidikan",
            16: "Uang Masuk",
            17: "Gaji",
            18: "Pencairan Investasi",
            19: "Bunga",
            20: "Refund",
            21: "Pencairan Pinjaman",
            22: "Cashback"
        }

        # mapping category
        df["category"] = df["category"].map(category_map)
        now = datetime.now()
        date_formated = now.strftime("%d-%m-%Y")
        # save data
        df.to_csv(f"data/raw_data/raw_data_{date_formated}.csv", index=False)

        print("Data ingestion executed")

    @task(task_id="pre_processing")
    def pre_processing():
        print("Executing data pre-processing")
        now = datetime.now()
        date_formated = now.strftime("%d-%m-%Y")

        # read .csv data
        df = pd.read_csv(f"data/raw_data/raw_data_{date_formated}.csv")

        # drop duplicates data
        df.drop_duplicates(subset=["merchantname"])

        # drop values with nan data
        df = df.dropna()

        # reindexing dataframe
        df.index = range(len(df))

        # lowercase merchant name for standardize (option can be use .title())
        df["merchantname"] = df["merchantname"].apply(lambda x: x.lower())

        df["merchantname"] = df["merchantname"].apply(lambda x: re.sub(r'/"[^"]*"/g', '', x))

        df.dropna()

        now = datetime.now()
        date_formated = now.strftime("%d-%m-%Y")
        # save data
        df.to_csv(f"data/pre-processed_data/pre-processed_data_{date_formated}.csv", index=False)
        
        print("Data pre-processing executed")

    @task(task_id="training_preparation")
    def training_preparation():
        print("Executing training preparation")
        now = datetime.now()
        date_formated = now.strftime("%d-%m-%Y")

        # read .csv data
        df = pd.read_csv(f"data/pre-processed_data/pre-processed_data_{date_formated}.csv")

        print(df["category"].unique())

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

        # create label column
        df["label"] = df.category.map(category_map)

        print(df["category"].value_counts())
        

        # splitting data
        X_train, X_val, y_train, y_val = train_test_split(df.index.values, 
                                                        df.label.values, 
                                                        test_size=0.10, 
                                                        random_state=42, 
                                                        stratify=df.label.values)
        df['data_type'] = ['not_set']*df.shape[0]

        df.loc[X_train, 'data_type'] = 'train'
        df.loc[X_val, 'data_type'] = 'val'

        # data splitting result
        print(df.groupby(['category', 'label', 'data_type']).count())

        # tokenize dataset
        tokenizer = BertTokenizer.from_pretrained('indolem/indobertweet-base-uncased', do_lower_case=True)

        encoded_data_train = tokenizer.batch_encode_plus(
            df[df.data_type=='train'].merchantname.values, 
            add_special_tokens=True, 
            return_attention_mask=True, 
            pad_to_max_length=True, 
            max_length=256, 
            return_tensors='pt'
        )

        encoded_data_val = tokenizer.batch_encode_plus(
            df[df.data_type=='val'].merchantname.values, 
            add_special_tokens=True, 
            return_attention_mask=True, 
            pad_to_max_length=True, 
            max_length=256, 
            return_tensors='pt'
        )


        input_ids_train = encoded_data_train['input_ids']
        attention_masks_train = encoded_data_train['attention_mask']
        labels_train = torch.tensor(df[df.data_type=='train'].label.values)

        input_ids_val = encoded_data_val['input_ids']
        attention_masks_val = encoded_data_val['attention_mask']
        labels_val = torch.tensor(df[df.data_type=='val'].label.values)

        dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
        dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)
        print(dataset_train)
        print(dataset_val)

        print("Training preparation executed")

    @task(task_id="hyperparameter_tuning")
    def hyperparameter_tuning():
        print("Executing hyperparameter tuning")
        optimum_params = {}
        return optimum_params

    @task(task_id="finetuning_model")
    def finetuning_model(optimum_params: dict):
        print("Executing finetuning model")

    @task(task_id="upload_model")
    def upload_model():
        print("Executing upload model")

    @task(task_id="evaluate_model")
    def evaluate_model():
        print("Executing evaluate model")

    @task(task_id="promote_model")
    def promote_model():
        print("Executing promote mdoel")

# data_ingestion_task = PythonOperator(
#  task_id='data_ingestion',
#  python_callable=data_ingestion,
#  dag=dag,
# )

# pre_processing_task = PythonOperator(
#  task_id='pre_processing',
#  python_callable=pre_processing,
#  dag=dag,
# )

# training_preparation_task = PythonOperator(
#  task_id='training_preparation',
#  python_callable=training_preparation,
#  dag=dag,
# )

# hyperparameter_tuning_task = PythonOperator(
#  task_id='hyperparameter_tuning',
#  python_callable=hyperparameter_tuning,
#  dag=dag,
# )

# finetuning_model_task = PythonOperator(
#  task_id='finetuning_model',
#  python_callable=finetuning_model,
#  dag=dag,
# )

# upload_model_task = PythonOperator(
#  task_id='upload_model',
#  python_callable=upload_model,
#  dag=dag,
# )

# promote_model_task = PythonOperator(
#  task_id='promote_model',
#  python_callable=promote_model,
#  dag=dag,
# )

# data_ingestion_task >> pre_processing_task >> training_preparation_task >> hyperparameter_tuning_task >> finetuning_model_task >> upload_model_task >> promote_model_task
    data_ingestion()
    pre_processing()
    training_preparation()
    params = hyperparameter_tuning()
    finetuning_model(params)
    upload_model()
    evaluate_model()
    promote_model()

# start DAG
dag_ml_pipeline = ml_pipeline()