from airflow.decorators import task

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tasks.models import MerchantGarage as ModelMerchantGarage
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import os

@task(task_id='Ingesting-Data')
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
    return df