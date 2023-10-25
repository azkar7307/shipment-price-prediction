import os, sys


ROOT_DIR = os.getcwd()

FILE_NAME = 'data.csv'

CONFIG_DIR = 'config'
SCHEMA_FILE = 'config.yaml'
CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, SCHEMA_FILE)

# Create variables related to our Data Ingestion Pipelinle

DATA_INGESTION_CONFIG_KEY = 'data_ingestion_config'
DATA_INGESTION_DATABASE_NAME = 'data_base'
DATA_INGESTION_COLLECTION_NAME = 'collection_name'
DATA_INGESTION_ARTIFACT_DIR = 'data_ingestion'
DATA_INGESTION_RAW_DATA_DIR_KEY = 'raw_data_dir'
DATA_INGESTION_INGESTED_DIR_KEY = 'ingested_dir'
DATA_INGESTION_TRAIN_DIR_KEY = 'ingested_train_dir'
DATA_INGESTION_TEST_DIR_KEY = 'ingested_test_dir'
CONFIG_FILE_KEY = 'config'

