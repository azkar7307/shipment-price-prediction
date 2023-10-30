import os, sys


ROOT_DIR = os.getcwd()

FILE_NAME = 'data.csv'

CONFIG_DIR = 'config'
# SCHEMA_FILE = 'config.yaml'
# CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, SCHEMA_FILE)

CONFIG_FILE = 'config.yaml'
CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, CONFIG_FILE)
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

# ********************Data Ingestion completed************************

# ********************Data Transformation started*********************

# Schema file path

ROOT_DIR = os.getcwd()
CONFIG_DIR = 'config'
SCHEMA_FILE = 'schema.yaml'
SCHEMA_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, SCHEMA_FILE)

# transformation file path

ROOT_DIR = os.getcwd()
CONFIG_DIR = 'config'
TRANSFORMATION_FILE = 'transformation.yaml'
TRANSFORMATION_YAML_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, TRANSFORMATION_FILE)

TARGET_COLUMN_KEY = 'target_column'
NUMERICAL_COLUMN_KEY = 'numerical_columns'
CATEGORICAL_COLUMN_KEY = 'categorical_columns'
OUTLIERS_COLUMN_KEY = 'outliers_columns'
DROP_COLUMN_KEY = 'drop_columns'
SCALING_COLUMN_KEY = 'scaling_columns'

# Data Transformation related variables keys
DATA_TRANSFORMATION_CONFIG_KEY = 'data_transformation_config'
DATA_TRANSFORMATION = 'data_transformation_dir'
DATA_TRANSFORMATION_DIR_NAME_KEY = 'transformed_dir'
DATA_TRANSFORMATION_TRAIN_DIR_NAME_KEY = 'transformed_train_dir'
DATA_TRANSFORMATION_TEST_DIR_NAME_KEY = 'transformed_test_dir'
DATA_TRANSFORMATION_PROCESSING_DIR_KEY = 'processing_dir'
DATA_TRANSFORMATION_PROCESSING_FILE_KEY = 'preprocessed_object_file_name'
DATA_TRANSFORMATION_FENG_FILE_KEY = 'feature_eng_file'

PICKLE_FOLDER_KEY_NAME = 'prediction_file'

# Prediction File path
ROOT_DIR = os.getcwd()
CONFIG_DIR = 'config'
PREDICTION_YAML_FILE = 'prediction.yaml'
PREDICTION_YAML_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, PREDICTION_YAML_FILE)