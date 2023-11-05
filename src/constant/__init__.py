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
DATA_TRANSFORMATION_CONFIG_KEY = 'data_transformation_config' # dict

                    # data_transformation
DATA_TRANSFORMATION = 'data_transformation_dir'

                                 # transformed_data
DATA_TRANSFORMATION_DIR_NAME_KEY = 'transformed_dir'

                                       # train
DATA_TRANSFORMATION_TRAIN_DIR_NAME_KEY = 'transformed_train_dir'

                                      # test
DATA_TRANSFORMATION_TEST_DIR_NAME_KEY = 'transformed_test_dir'

                                        # processed
DATA_TRANSFORMATION_PROCESSING_DIR_KEY = 'processing_dir'

                                        # processed.pkl
DATA_TRANSFORMATION_PROCESSING_FILE_KEY = 'preprocessed_object_file_name'

                                    # feature_eng.pkl
DATA_TRANSFORMATION_FENG_FILE_KEY = 'feature_eng_file'

PICKLE_FOLDER_KEY_NAME = 'prediction_file'

# Prediction File path
ROOT_DIR = os.getcwd()
CONFIG_DIR = 'config'
PREDICTION_YAML_FILE = 'prediction.yaml'
PREDICTION_YAML_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, PREDICTION_YAML_FILE)


# ******************** Data Transformation completed *********************

# ******************** Model Training started *********************


# Model Training

MODEL_TRAINING_CONFIG_KEY = 'model_trainer_config'
MODEL_TRAINING_ARTIFACT_DIR = 'model_training_dir'
MODEL_TRAINING_OBJECT = 'model_object_file'
MODEL_REPORT_FILE = 'model_report_file'

# Saved Model

SAVED_MODEL_CONFIG_KEY = 'saved_model_config'
SAVED_MODEL_DIR = 'saved_model_dir'
SAVED_MODEL_OBJECT = 'model_object_file' 
SAVED_MODEL_REPORT = 'model_report_file'








