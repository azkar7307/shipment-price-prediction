from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    train_file_path:str
    test_file_path:str

# ********************Data Ingestion completed************************

# ********************Data Transformation started*********************

@dataclass
class DataTransformationArtifact:
    transformed_train_file_path:str
    train_target_file_path:str
    transformed_test_file_path:str
    test_target_file_path:str    
    feature_eng_obj_file_path:str

# ******************** Data Transformation completed *********************

# ******************** Model Training started *********************

@dataclass
class ModelTrainerArtifact:
    trained_model_file_path:str
    model_artifact_report:str
