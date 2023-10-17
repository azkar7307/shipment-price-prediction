import pandas as pd
import pymongo
import json
# from pymongo.mongo_client import MongoClient

client = "mongodb+srv://azkar7307:createacluster@cluster0.scidade.mongodb.net/?retryWrites=true&w=majority"


DATA_FILE_PATH = (r'D:\MLOps\shipment-price-prediction\data\train.csv')
DATABASE = 'machine_learning'
COLLECTION_NAME = 'DATASET'

if __name__ == '__main__':
    df = pd.read_csv(DATA_FILE_PATH)
    print(f'Rows and Columns of our Data: {df.shape}')

    df.reset_index(drop=True, inplace=True)

    json_record = list(json.loads(df.T.to_json()).values())

    print(json_record[0])

    client[DATABASE][COLLECTION_NAME].insert_many(json_record)










