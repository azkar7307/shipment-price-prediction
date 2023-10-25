import os, sys
from src.logger import logging
from src.exception import CustomException
from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataIngestionConfig
import pandas as pd
from src.utils import get_collection_as_dataframe
import shutil
from src.constant import *
from src.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig
from src.components.data_ingestion import DataIngestion

class Pipeline():
    def __init__(self, training_pipeline_config = TrainingPipelineConfig())-> None:
        try:
            self.training_pipeline_config = training_pipeline_config
        except Exception as e:
            raise CustomException(e, sys) # from e
        

    def start_data_ingestion(self)->DataIngestionArtifact:

        try:
            data_ingestion = DataIngestion(data_ingestion_config=DataIngestionConfig(self.training_pipeline_config))

            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise CustomException(e, sys) # from e
        

    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
        except Exception as e:
            raise CustomException(e, sys) # from e