import os, sys
from src.exception import CustomException
from src.logger import logging
from datetime import datetime
# from src.constant import *
from src.utils import read_yaml_file

# Data Ingestion
from src.constant import CONFIG_FILE_PATH
from src.constant import DATA_INGESTION_CONFIG_KEY
from src.constant import DATA_INGESTION_DATABASE_NAME, DATA_INGESTION_COLLECTION_NAME
from src.constant import DATA_INGESTION_ARTIFACT_DIR
from src.constant import DATA_INGESTION_CONFIG_KEY
from src.constant import DATA_INGESTION_RAW_DATA_DIR_KEY
from src.constant import DATA_INGESTION_INGESTED_DIR_KEY
from src.constant import DATA_INGESTION_TRAIN_DIR_KEY
from src.constant import DATA_INGESTION_TEST_DIR_KEY

# Data Transformation
from src.constant import DATA_TRANSFORMATION_CONFIG_KEY
from src.constant import DATA_TRANSFORMATION
from src.constant import DATA_TRANSFORMATION_DIR_NAME_KEY
from src.constant import DATA_TRANSFORMATION_TRAIN_DIR_NAME_KEY
from src.constant import DATA_TRANSFORMATION_TEST_DIR_NAME_KEY
from src.constant import DATA_TRANSFORMATION_PROCESSING_DIR_KEY
from src.constant import DATA_TRANSFORMATION_PROCESSING_FILE_KEY
from src.constant import DATA_TRANSFORMATION_FENG_FILE_KEY

# Model Trainer
from src.constant import MODEL_TRAINING_CONFIG_KEY
from src.constant import MODEL_TRAINING_ARTIFACT_DIR
from src.constant import MODEL_TRAINING_OBJECT
from src.constant import MODEL_REPORT_FILE

from src.constant import SAVED_MODEL_CONFIG_KEY
from src.constant import SAVED_MODEL_DIR



# D:\MLOps\shipment-price-prediction\config\config.yaml
config_data = read_yaml_file(CONFIG_FILE_PATH)

class TrainingPipelineConfig:
    def __init__(self):
        try:
            self.artifact_dir = os.path.join(os.getcwd(), 'artifact', f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        except Exception as e:
            raise CustomException(e, sys)
        

class DataIngestionConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        try:
            data_ingestion_key = config_data[DATA_INGESTION_CONFIG_KEY]
            logging.info('Setting the data ingestion key')

            self.databse_name = data_ingestion_key[DATA_INGESTION_DATABASE_NAME]
            self.collection_name = data_ingestion_key[DATA_INGESTION_COLLECTION_NAME]
            logging.info('Setting data ingestion database and collection')

            self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir, data_ingestion_key[DATA_INGESTION_ARTIFACT_DIR])
            self.raw_data_dir = os.path.join(self.data_ingestion_dir, data_ingestion_key[DATA_INGESTION_RAW_DATA_DIR_KEY])
            self.ingested_data_dir = os.path.join(self.raw_data_dir, data_ingestion_key[DATA_INGESTION_INGESTED_DIR_KEY])

            logging.info('Training file path is creating')
            self.train_file_path = os.path.join(self.ingested_data_dir, data_ingestion_key[DATA_INGESTION_TRAIN_DIR_KEY])
            self.test_file_path = os.path.join(self.ingested_data_dir, data_ingestion_key[DATA_INGESTION_TEST_DIR_KEY])

            # myEdit
            self.raw_data_file_path = os.path.join(self.raw_data_dir, data_ingestion_key[DATA_INGESTION_RAW_DATA_DIR_KEY])

            self.test_size = 0.2
        
        except Exception as e:
            raise CustomException(e, sys)

# ********************Data Ingestion completed************************

# ********************Data Transformation started*********************

class DataTransformationConfig:
    
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        
        # dict
        data_transformation_key = config_data[DATA_TRANSFORMATION_CONFIG_KEY]
        
        # data_transformation
        # D:\MLOps\shipment-price-prediction\artifact\2023-10-31_11-06-28\data_transformation
        self.data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir, data_transformation_key[DATA_TRANSFORMATION])
        
        # transformed_data
        # D:\MLOps\shipment-price-prediction\artifact\2023-10-31_11-06-28\data_transformation\transformed_data
        self.transformation_dir = os.path.join(self.data_transformation_dir, data_transformation_key[DATA_TRANSFORMATION_DIR_NAME_KEY])

        # train
        # D:\MLOps\shipment-price-prediction\artifact\2023-10-31_11-06-28\data_transformation\transformed_data\train
        self.transformed_train_dir = os.path.join(self.transformation_dir, data_transformation_key[DATA_TRANSFORMATION_TRAIN_DIR_NAME_KEY])

        # test
        # D:\MLOps\shipment-price-prediction\artifact\2023-10-31_11-06-28\data_transformation\transformed_data\test
        self.transformed_test_dir = os.path.join(self.transformation_dir, data_transformation_key[DATA_TRANSFORMATION_TEST_DIR_NAME_KEY])

        # processed
        # # D:\MLOps\shipment-price-prediction\artifact\2023-11-02_20-18-23\data_transformation\transformed_data\processed
        # self.processing_dir = os.path.join(self.transformation_dir, data_transformation_key[DATA_TRANSFORMATION_PROCESSING_DIR_KEY])
        
        # processed.pkl
        # D:\MLOps\shipment-price-prediction\artifact\2023-11-02_20-18-23\data_transformation\transformed_data\processed\processed.pkl
        # self.processor_object_file = os.path.join(self.processing_dir, data_transformation_key[DATA_TRANSFORMATION_PROCESSING_FILE_KEY])
        
        # feature_eng.pkl
        # D:\MLOps\shipment-price-prediction\artifact\2023-10-31_11-06-28\data_transformation\transformed_data\processed\feature_eng.pkl
        # self.feature_engineering_dir = os.path.join(self.processing_dir, data_transformation_key[DATA_TRANSFORMATION_FENG_FILE_KEY])


        # edit
        # processed
        # D:\MLOps\shipment-price-prediction\artifact\2023-11-02_20-18-23\data_transformation\processed
        self.processing_dir = os.path.join(self.data_transformation_dir, data_transformation_key[DATA_TRANSFORMATION_PROCESSING_DIR_KEY])

        # processed.pkl        
        # D:\MLOps\shipment-price-prediction\artifact\2023-10-31_11-06-28\data_transformation\processed\processed.pkl
        self.processor_object_file = os.path.join(self.processing_dir, data_transformation_key[DATA_TRANSFORMATION_PROCESSING_FILE_KEY])
        
        # feature_eng.pkl
        # D:\MLOps\shipment-price-prediction\artifact\2023-10-31_11-06-28\data_transformation\processed\feature_eng.pkl
        self.feature_engineering_dir = os.path.join(self.processing_dir, data_transformation_key[DATA_TRANSFORMATION_FENG_FILE_KEY])
    

# ******************** Data Transformation completed *********************

# ******************** Model Training started *********************

class ModelTrainingConfig:
    
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        
        model_training_key = config_data[MODEL_TRAINING_CONFIG_KEY]
        
        self.model_training_dir = os.path.join(training_pipeline_config.artifact_dir, model_training_key[MODEL_TRAINING_ARTIFACT_DIR])
        
        self.model_object_file_path = os.path.join(self.model_training_dir, model_training_key[MODEL_TRAINING_OBJECT])
        self.model_report_file_path = os.path.join(self.model_training_dir, model_training_key[MODEL_REPORT_FILE])

        logging.info('All config is working fine')


class SavedModelConfig:

    def __init__(self):

        saved_model_config = config_data[SAVED_MODEL_CONFIG_KEY]
        ROOT_DIR = os.getcwd()
        self.saved_model_dir = os.path.join(ROOT_DIR, saved_model_config[SAVED_MODEL_DIR])