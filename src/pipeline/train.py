import os, sys
from src.logger import logging
from src.exception import CustomException
from src.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact, ModelTrainerArtifact
from src.entity.config_entity import DataIngestionConfig, DataTransformationConfig
import pandas as pd
from src.utils import get_collection_as_dataframe
import shutil
from src.constant import *
from src.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, ModelTrainingConfig
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer

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
        
    def start_data_transformation(self, data_ingestion_artifact=DataIngestionArtifact) -> DataTransformationArtifact:
        try:
            data_transformation = DataTransformation(
                data_transformation_config = DataTransformationConfig(self.training_pipeline_config),
                data_ingestion_artifact = data_ingestion_artifact
            )

            return data_transformation.initiate_data_transformation()
        except Exception as e:
            raise CustomException(e, sys) from e

    def start_model_training(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        
        try:
            model_training = ModelTrainer(model_training_config=ModelTrainingConfig(self.training_pipeline_config),
                                           data_transformation_artifact = data_transformation_artifact)
            
            logging.info("Model training Initialized")

            return model_training.start_model_training()
        
        except Exception as e:
            raise CustomException(e, sys)


    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()

            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact)

            model_trainer_artifact = self.start_model_training(data_transformation_artifact= data_transformation_artifact)
            
        except Exception as e:
            raise CustomException(e, sys) # from e