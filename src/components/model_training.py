import os, sys
import yaml
from src.exception import CustomException
from src.logger import logging
from datetime import datetime
import numpy as np
import pandas as pd
from src.entity.config_entity import ModelTrainingConfig
from src.entity.config_entity import SavedModelConfig
from src.entity.artifact_entity import DataTransformationArtifact
from src.utils import load_numpy_array_data, save_object, read_yaml_file




from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import optuna

# Hyperparameter tunning
# Trainer class
# model training

# XGBRegressor("n_estimator": [100, 200, 300, 400, 500])
# XGBRegressor("max_split": [100, 200, 300, 400, 500])
# XGBRegressor("min_split": [100, 200, 300, 400, 500])
# XGBRegressor("n_estimator": [100, 200, 300, 400, 500])

class OptunaTuner:
    
    def __init__(self, model, params, X_train, y_train, X_test, y_test):
        self.model = model
        self.params = params
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
    
    def Objective(self, trial):
        
        params_values = {}
        for key, value_range in self.params.items():
            if value_range[0] <= value_range[1]:
                if isinstance(value_range[0], int) and isinstance(value_range[1], int):
                    params_values[key] = trial.suggest_int(key, value_range[0], value_range[1])
                else:
                    params_values[key] = trial.suggest_float(key, value_range[0], value_range[1])
            else:
                raise ValueError(f'Invalid range for {key}: low = {value_range[0]}, high={value_range[1]}')
        
        self.model.set_params(**params_values)

        # Fit the model on the training_data
        self.model.fit(self.X_train, self.y_train)

        # Predict on the test data
        y_pred = self.model.predict(self.X_test)

        # Calculate the r2 Score
        r2 = r2_score(self.y_test, y_pred)

        return r2
    

    def tune(self, n_trials = 100):
        
        study = optuna.create_study(direction='maximize') # maximize R2 Score
     
        study.optimize(self.Objective, n_trials=n_trials)
        best_params = study.best_params
        print(f'best parameters: {best_params}')

        # Set the best parameter to the our model
        self.model.set_params(**best_params)

        # Retrain the model with best parameters on the training set
        self.model.fit(self.X_train, self.y_train)

        # Evaluate the model on the test data using R2 Score
        y_pred_test = self.model.predict(self.X_test)

        best_r2_score = r2_score(self.y_test, y_pred_test)

        print(f'best R2 Score on test Sets: {best_r2_score}')

        return best_r2_score, self.model, best_params


class trainer:

    def __init__(self) -> None:
        self.model_dict = {

            'Random_Forest_Regressor': RandomForestRegressor(), 
            'Gradient_Boost_Regressor': GradientBoostingRegressor(),
            'XGBRegressor': xgb.XGBRegressor()

        }

        self.param_dict = {
            'Random_Forest_Regressor': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 10],
                'min_samples_split': [2, 10, 50],
                'min_samples_leaf': [1, 4, 25]
            },

            'Gradient_Boost_Regressor': {
                'n_estimators': [100, 300, 500],
                'max_depth': [3, 5, 10],
                'learning_rate': [0.01, 0.1, 0.5, 0.9]
            }, 

            'XGBRegressor': {
                'n_estimators': [100, 300, 500],
                'max_depth': [3, 5, 10],
                'learning_rate': [0.01, 0.1, 0.5, 0.9],
                'min_sample_split': [2, 10, 50],
                'min_sample_leaf': [1, 4, 25]
            
            }

        }


class ModelTrainer:

    def __init__(self, model_training_config: ModelTrainingConfig,
                    data_transformation_artifact: DataTransformationArtifact):
        
        try:
            self.model_training_config = model_training_config
            self.data_transformation_artifact = data_transformation_artifact

            self.saved_model_config = SavedModelConfig()
            self.saved_model_dir = self.saved_model_config.saved_model_dir

        except Exception as e:
            raise CustomException(e, sys)
    
    def start_model_training(self):

        try:

            X_train = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            y_train = load_numpy_array_data(self.data_transformation_artifact.train_target_file_path)
            X_test = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)
            y_test = load_numpy_array_data(self.data_transformation_artifact.test_target_file_path)

            models = trainer()

            results = {} # model_name: 60

            tuned_models = []

            for model_name, model in models.model_dict.items():

                logging.info(f"Tuning and fitting model ----------->>>>  {model_name}")
                
                # Create an instance of OptunaTuner for each model
                tuner = OptunaTuner(model, params=models.param_dict[model_name], X_train=X_train, y_train=y_train.ravel(), 
                                                                                        X_test=X_test, y_test=y_test.ravel())
            
                # Perform hyperparameter tuning

                best_r2_score, tunned_model, best_params = tuner.tune(n_trials=5)

                logging.info(f"Best R2 score for {model_name}: {best_r2_score}")
                logging.info("----------------------")

                # Append the tuned model to the list of tuned models
                tuned_models.append((model_name, tunned_model))

                results[model_name] = best_r2_score

            # Convert the results dict to a dataframe
            result_df = pd.DataFrame(results.items(), columns=['model', 'R2_Score'])
            # Sort the DataFrame by 'R2_Score' in descending order
            result_df_sorted = result_df.sort_values(by='R2_Score', ascending=False)

            # Get the best model (the one with the highest R2 score)
            best_model_name = result_df_sorted.iloc[0]['model']

            # Iterate through the list and look for the desired model
            for model_tuple in tuned_models:
                if model_tuple[0] == best_model_name:
                    best_model = model_tuple[1]
                    break # Exit the loop once you've found the desired model

            best_r2_score = result_df_sorted.iloc[0]['R2_Score']
            logging.info("-------------")
            os.makedirs(self.saved_model_dir, exist_ok=True)

            contents = os.listdir(self.saved_model_dir)

            logging.info(f"The values in contents: {contents}")
            
            artifact_model_score = best_r2_score


            if not contents:
                # Model Report
                model_report = {'Model_name': best_model_name,
                                'R2_Score': str(best_r2_score),
                                'parameters': best_params}
                
                logging.info(f"Model Report: {model_report}")
                
                file_path = os.path.join(self.saved_model_dir, 'model.pkl')
                save_object(file_path = file_path, obj = best_model)
                logging.info("Model saved.")

                # Saved yaml file

                file_path = os.path.join(self.saved_model_dir, 'report.yaml')
                with open(file_path, 'w') as file:
                    yaml.dump(model_report, file)
                
                logging.info("Report saved as YAML file.")
            
            
            
            else:
                # Saved model data
                report_file_path = os.path.join(self.saved_model_dir, 'report.yaml')
                saved_model_report_data = read_yaml_file(file_path=report_file_path)

                # Model Trained artifact data
 
                artifact_model_score = best_r2_score
                saved_model_score = float(saved_model_report_data['R2_Score'])
            
                # Model Registry -> 10 (1 0.90)
                # Trained_model_dir / Saved_model_dir (1 0.82) -> Removed

                if artifact_model_score > saved_model_score:
                    model_report = {'model_name': best_model_name, 
                                    'R2_Score': str(best_r2_score),
                                    'parameter': best_params}

                    logging.info(f"Model Report: {model_report}")

                    file_path = os.path.join(self.saved_model_dir, 'model.pkl')
                    save_object(file_path=file_path, obj = best_model)
                    logging.info("Model saved.")

                    # Save the report as a YAML file
                    file_path = os.path.join(self.saved_model_dir, 'report.yaml')
            
                    with open(file_path, 'w') as file:
                        yaml.dump(model_report, file)
                
                    logging.info('Report Saved as YAML File')
                
                
                else:
                    logging.info('Saved model in the directory is better than the trained model')


        except Exception as e:
            raise CustomException(e, sys)







