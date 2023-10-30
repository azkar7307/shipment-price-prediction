import os, sys
from src.exception import CustomException
from src.logger import logging
from datetime import datetime
from src.constant import *
from src.utils import read_yaml_file

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
        
        data_transformation_key = config_data[DATA_TRANSFORMATION_CONFIG_KEY]

        self.data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir, data_transformation_key[DATA_TRANSFORMATION])
        self.transformation_dir = os.path.join(self.data_transformation_dir, data_transformation_key[DATA_TRANSFORMATION_DIR_NAME_KEY])
        self.transformed_train_dir = os.path.join(self.transformation_dir, data_transformation_key[DATA_TRANSFORMATION_TRAIN_DIR_NAME_KEY])
        self.transformed_test_dir = os.path.join(self.transformation_dir, data_transformation_key[DATA_TRANSFORMATION_TEST_DIR_NAME_KEY])

        self.processing_dir = os.path.join(self.transformation_dir, data_transformation_key[DATA_TRANSFORMATION_PROCESSING_DIR_KEY])
        self.processor_object_file = os.path.join(self.processing_dir, data_transformation_key[DATA_TRANSFORMATION_PROCESSING_FILE_KEY])
        self.feature_engineering_dir = os.path.join(self.processing_dir, data_transformation_key[DATA_TRANSFORMATION_FENG_FILE_KEY])
    

