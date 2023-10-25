import pandas as pd
import json
import os, sys
from dotenv import load_dotenv
import pymongo


def mongodb_client():
    ROOT_DIR = os.getcwd()
    env_file_path = os.path.join(ROOT_DIR, '.env')

    load_dotenv(dotenv_path = env_file_path)

    username = os.getenv('USER_NAME')
    password = os.getenv('PASS_WORD')
    cluster_name = os.getenv('CLUSTER_LABEL')

    mongo_db_url = f"mongodb+srv://{username}:{password}@{cluster_name}.mongodb.net/?retryWrites=true&w=majority"

    print(mongo_db_url)

    client = pymongo.MongoClient(mongo_db_url)
    
    return client