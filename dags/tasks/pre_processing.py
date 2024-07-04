import pandas as pd
from datetime import datetime
import re
from dotenv import load_dotenv

from airflow.decorators import task


@task(task_id='Preprocessing-data')
def pre_processing(data):
    load_dotenv()
    print("Executing data pre-processing")
    # drop duplicates data
    data.drop_duplicates(subset=["merchantname"])

    # drop values with nan data
    data = data.dropna()

    # reindexing dataframe
    data.index = range(len(data))

    # lowercase merchant name for standardize (option can be use .title())
    data["merchantname"] = data["merchantname"].apply(lambda x: x.lower())

    data["merchantname"] = data["merchantname"].apply(lambda x: re.sub(r'/"[^"]*"/g', '', x))

    data.dropna()

    now = datetime.now()
    date_formated = now.strftime("%d-%m-%Y")
    # save data
    data.to_csv(f"data/pre-processed_data/pre-processed_data_{date_formated}.csv", index=False)
    
    print("Data pre-processing executed")
    return data
