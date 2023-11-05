import os, sys
from src.logger import logging
from src.exception import CustomException
from src.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact
from src.entity.config_entity import DataIngestionConfig
import pandas as pd
import numpy as np
from src.utils import get_collection_as_dataframe
import shutil
from sklearn.model_selection import train_test_split
from src.constant import *
from src.constant import TARGET_COLUMN_KEY
from src.utils import read_yaml_file, create_yaml_file_numerical_columns, create_yaml_file_categorical_columns_from_dataframe, save_numpy_array_data, save_object
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# Reading Data in Transformation Schema file
transformation_yaml = read_yaml_file(file_path=TRANSFORMATION_YAML_FILE_PATH)

# Column data accessed from schema.yaml 
target_column = transformation_yaml[TARGET_COLUMN_KEY] 
numerical_columns = transformation_yaml[NUMERICAL_COLUMN_KEY]
categorical_columns = transformation_yaml[CATEGORICAL_COLUMN_KEY]
drop_columns = transformation_yaml[DROP_COLUMN_KEY]

# Transformation
outlier_columns = transformation_yaml[OUTLIERS_COLUMN_KEY]
scaling_columns = transformation_yaml[SCALING_COLUMN_KEY]

class Feature_engineering(BaseEstimator, TransformerMixin):
    #1. handling missing data
    #2. Drop Columns
    #3. Drop columns if nan above > 70%
    #4. Handle Outliers / Trim Outliers
    #5. Remove Outliers
    #6. Handle Categorical Data
    #7. Transform our Data
    #8. Handle Datetime data

    def __init__(self):

        '''
        This class applies necessary Feature Engineering 
        '''
        logging.info(f'\n{"*"*20}Feature Engineering Started{"*"*20}\n')
        logging.info(f' Numerical Columns, Categorical Columns, Target Column initialised in Feature Engineering Pipeline')
    
    # Feature Engineering Pipeline
    
            # ############################## Data Modification ############################## 


    def df_drop_columns(self, X:pd.DataFrame) -> pd.DataFrame:

        if 'Unnamed:_0' in X.columns:
            X = X.drop(columns=['Unnamed:_0'])
            logging.info('Extra columns called "Unnamed: 0" found in dataframe and has been dropped from df')
        
        columns_to_drop = drop_columns
        logging.info(f'Dropping Columns : {columns_to_drop}')
        logging.info(f'inside df_drop_columns(), the columns in X are:{X.columns}')

        X.drop(columns=columns_to_drop, inplace=True)

        return X
    
    def replace_spaces_with_underscore(self, df):
        logging.info(f'Columns name befor modified:\n {df.columns}')
        df = df.rename(columns = lambda x: x.replace(' ', '_'))
        logging.info(f'Columns name after modified:\n {df.columns}')

        return df

    def replace_nan_with_random(self, df, column_label):
        if column_label not in df.columns:
            print(f'Within replace_nan_with_random(), the Column "{column_label}" not found in the DataFrame')

            return df
        
        original_data = df[column_label].copy()
        nan_indices = df[df[column_label].isna()].index
        num_nan = len(nan_indices)

        existing_values = original_data.dropna().values
        random_values = np.random.choice(existing_values, num_nan)
        df.loc[nan_indices, column_label] = random_values

        original_mean = original_data.mean()
        original_median = original_data.median()
        new_mean = df[column_label].mean()
        new_median = df[column_label].median()

        return df
    
    def drop_rows_with_nan(self, X: pd.DataFrame, column_label: str):

        # Log the shape before dropping Nan values
        logging.info(f'Shape before dropping NaN values: {X.shape}')
        
        # Drop rows with Nan values in the specified column
        X = X.dropna(subset=[column_label])
        #X.to_csv('NaN values_removed.csv', index=False)

        # Log the shape after dropping the NaN values
        X = X.reset_index(drop=True)
        logging.info(f'Shape after dropping NaN values {X.shape}')

        return X
    
    def trim_outliers_by_quantile(self, df, column_label, upper_quantile=0.95, lower_quantile=0.05):

        if column_label not in df.columns:
            print(f'Within trim_outliers_by_quantile(), Column "{column_label}" not in DataFrame')
            return df
        
        column_data = df[column_label]

        lower_bound = column_data.quantile(lower_quantile)
        upper_bound = column_data.quantile(upper_quantile)

        trimmed_data = column_data.clip(lower=lower_bound, upper = upper_bound)
        df[column_label] = trimmed_data
        
        return df
    
    def remove_outliers(self, X):
        for column in outlier_columns:
            logging.info(f'Within remove_outliers():: Removing Outlier from column: {column}')
            X = self.trim_outliers_by_quantile(df = X, column_label = column)
        
        return X
    
    def run_data_modification(self, data):

        X = data.copy()

        logging.info(' Editing Column Labels ... ')
        logging.info(f'outliers_columns: {outlier_columns}')
        X = self.replace_spaces_with_underscore(X)

        try:
            X = self.df_drop_columns(X)
        
        except Exception as e:
            print('Test Data does not consists of some Dropped Columns')
        
        logging.info('---------------------------')
        logging.info('Replace nan with random data')
        for column in ['Artist_Reputation', 'Height', 'Width']:
            # Removing nan rows
            logging.info(f'Removing NaN values from the column: {column}')
#
            X = self.replace_nan_with_random(X, column_label=column)

        logging.info('-----------------------------')
        logging.info('Dropping rows with nan')
        for column in ['Weight', 'Material', 'Remote_Location']:
            # Removing nan rows
            logging.info(f'Dropping rows from column: {column}')
            X = self.drop_rows_with_nan(X, column_label=column)
        
        X = self.remove_outliers(X)

        return X
    
    
    def data_wrangling(self, X:pd.DataFrame):

        try:
            # Data Modification
            data_modified = self.run_data_modification(data = X)
            logging.info('Data modification done!')

            return data_modified
        
        except Exception as e:
            raise CustomException(e, sys) from e

    
    def fit(self, X, y=None):
        
        return self
    
    def transform(self, X:pd.DataFrame, y = None):

        try:
            data_modified = self.data_wrangling(X)

            #data_modified.to_csv('data_modified.csv', index=False)
            logging.info("Data Wrangling done!")

            logging.info(f'Original Data: {X.shape}')
            logging.info(f'Shape Modified Data: {data_modified.shape}')
            
            return data_modified
        
        except Exception as e:
            raise CustomException(e, sys) from e


#
class DataProcessor:
    def __init__(self, numerical_cols, categorical_cols):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols

 
        # Define preprocessing steps using a Pipeline
        categorical_transformer = Pipeline(
            steps = [
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]
        )

        numerical_transformer = Pipeline(
            steps = [
                ('log_transform', FunctionTransformer(np.log1p , validate=False))
            ]
        )

        # Create a ColumnsTransformer to apply the transformation
        self.preprocessor = ColumnTransformer(
            transformers = [
                ('cat', categorical_transformer, categorical_cols),
                ('num', numerical_transformer, numerical_cols)
            ],
            remainder='passthrough'
        )


    def get_preprocessor(self):
        return self.preprocessor
        
    def fit_transform(self, data):
#
        # fit and transform the data using the preprocessor
        transformed_data = self.preprocessor.fit_transform(data)

        return transformed_data
        


class DataTransformation:
    def __init__(self, data_transformation_config: DataIngestionConfig, 
                 data_ingestion_artifact: DataIngestionArtifact):
        
        try:
            logging.info(f'\n{"*"*20}Data Transformation log Started{"*"*20}\n')
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact

        except Exception as e:
            raise CustomException(e, sys) from e

    def get_feature_engineering_object(self):
        try:
            feature_engineering = Pipeline(steps = [('fe', Feature_engineering())])

            return feature_engineering

        except Exception as e:
            raise CustomException(e, sys) from e


    def seperate_numerical_categorical_columns(self, df):
        numerical_columns = df.select_dtypes(include=np.number).columns.tolist()
        categorical_columns = df.select_dtypes(include=[object, pd.Categorical]).columns.tolist()

        return numerical_columns, categorical_columns

    def initiate_data_transformation(self):

        try:
            # Data validation Artifact -----------> Accessing train and test files
            logging.info(f'Obtaining training and test file path')
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            logging.info('Loading train and test data as pandas dataframe.')
            logging.info(f' Accessing train file from: {train_file_path}\
                            Test File Path:{test_file_path}')
            train_df = pd.read_csv(train_file_path)
            test_df = pd.read_csv(test_file_path)
            logging.info('Successfully Loaded the train and the test data into pandas dataframe.')
            
            logging.info(f'Columns in train_df: {train_df.columns}')
            logging.info(f'Columns in test_df: {test_df.columns}')
    
            # logging.info(f'after the pd.read_csv, train_columns are:\n{train_df.columns}')
            # logging.info(f'after the pd.read_csv, test_columns are:\n{test_df.columns}')

            # train_df = train_df.drop(columns='Unnamed: 0')
            # test_df = test_df.drop(columns='Unnamed: 0')

            # logging.info(f'after the modification in columns, the train_columns are:\n{train_df.columns}')
            # logging.info(f'after the modification in columns, the test_columns are:\n{test_df.columns}')
            
            logging.info(f'Target Column: {target_column}')
            logging.info(f'Numerical Columns: {numerical_columns}')
            logging.info(f'Categorical Columns: {categorical_columns}')
            logging.info(f'drop Columns: {drop_columns}')

            all_cols = numerical_columns + categorical_columns + drop_columns + target_column
            all_cols = list(set(all_cols))
            # All columns
            logging.info(f'All columns: {all_cols}')
            logging.info(f'train_df columns not in "all_cols": {set(train_df.columns).difference(all_cols)}')

            # Feature Engineering
            logging.info('Obtaining feature engineering object.')
            fe_obj = self.get_feature_engineering_object()

            logging.info(f'Applying feature engineering object on training dataframe')
            logging.info('>>>' * 20 + 'Training data ' + '<<<' * 20)
            
            train_df = fe_obj.fit_transform(X=train_df)
            logging.info(f'Applied feature engineering object on training dataframe')
            logging.info('train df info after applying FE:\n')
            logging.info(f'columns: {train_df.columns}')
            logging.info(f'shape: {train_df.shape}')
            logging.info(f'dtypes: {set(train_df.dtypes.values)}')
            logging.info(f'null_counts: {train_df.isnull().sum().sum()}')
            
            logging.info(f'Applying feature engineering object on testing datafrme')
            logging.info('>>>' * 20 + 'Test data ' + '<<<' * 20)
            logging.info(f'Feature Engineering - Test Data')
            test_df = fe_obj.transform(X=test_df)
            logging.info(f'Applied feature engineering object on testing datafrme')
            logging.info('test df info after applying FE:\n')
            logging.info(f'columns: {test_df.columns}')
            logging.info(f'shape: {test_df.shape}')
            logging.info(f'dtypes: {set(test_df.dtypes.values)}')
            logging.info(f'null_counts: {test_df.isnull().sum().sum()}')

            # Train Data
            logging.info('Feature Engineering with train and test data completed.')
            feature_eng_train_df:pd.DataFrame = train_df.copy() 
        #   feature_eng_train_df.to_csv('feature_eng_train_df.csv')
            logging.info(f' Columns in feature engineering train: {feature_eng_train_df.columns}')
            logging.info('Feature Engineering - Train Completed')

            # Test Data
            feature_eng_test_df:pd.DataFrame = test_df.copy() # Complete feature engineering
        #   feature_eng_test_df.to_csv('feature_eng_test_df.csv')
            logging.info(f' Columns in feature engineering test: {feature_eng_test_df.columns}')
            logging.info('Saving feature engineering training and testing dataframe.')
            logging.info('Feature Engineering - Train Completed')
            
            # Getting numerical and categorical of Transformed data

            # Train and test
#
            input_feature_train_df = feature_eng_train_df.drop(columns = target_column)# , axis=1)
            train_target_array = feature_eng_train_df[target_column]
#
            input_feature_test_df = feature_eng_test_df.drop(columns = target_column) # , axis=1)
            test_target_array = feature_eng_test_df[target_column]

                        # ############## Input Feature Transformation ##############
            
            ### Preprocessing
            logging.info('*' * 20 + 'Applying preprocessing object on training dataframe and testing dataframe ' + '*' * 20)

            logging.info(f' Scaling columns: {scaling_columns}')

            # Transforming Data
            logging.info(f'input_feature_train_df Columns\n: {input_feature_train_df.columns}')
            numerical_cols, categorical_cols = self.seperate_numerical_categorical_columns(df = input_feature_train_df)

#
            # Saving column labels for prediction
            create_yaml_file_numerical_columns(column_list= numerical_cols, yaml_file_path=PREDICTION_YAML_FILE_PATH)

            create_yaml_file_categorical_columns_from_dataframe(dataframe=input_feature_train_df, 
                                                                categorical_columns=categorical_cols, 
                                                                yaml_file_path=PREDICTION_YAML_FILE_PATH)


            logging.info(f' Transformed Data Numerical Columns: {numerical_cols}')
            logging.info(f' Transformed Data Categorical Columns: {categorical_cols}')

            # Setting columns in order

            column_order = numerical_cols + categorical_cols

            logging.info(f'Column_orders columns: {column_order}')

            input_feature_train_df = input_feature_train_df[column_order]
            input_feature_test_df = input_feature_test_df[column_order]

            data_preprocessor = DataProcessor(numerical_cols=numerical_cols, categorical_cols=categorical_cols)

            preprocessor = data_preprocessor.get_preprocessor()

            transformed_train_array = preprocessor.fit_transform(X=input_feature_train_df)

            train_target_array = train_target_array 

            transformed_test_array = preprocessor.transform(X=input_feature_test_df)

            test_target_array = test_target_array

            # Log the shape of transformed data

            logging.info('----------------------Transformed Data----------------------')
#
            # transformed_dir = self.data_transformation_config.transformation_dir
            # os.makedirs(transformed_dir, exist_ok=True)

            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir
# ==============================================================================================

            # transformed_train_dir = os.path.join(transformed_dir, transformed_train_dir)
            # transformed_test_dir = os.path.join(transformed_dir, transformed_test_dir)

# =========================================================================================
            
            os.makedirs(transformed_train_dir, exist_ok=True)
            os.makedirs(transformed_test_dir, exist_ok=True)

            # Transformed train and tests file

            logging.info('Saving Transformed Training and Transformed test data')
            transformed_train_file_path = os.path.join(transformed_train_dir, 'train.npz')
            train_target_file_path = os.path.join(transformed_train_dir, 'train_target.npz')
            transformed_test_file_path = os.path.join(transformed_test_dir, 'test.npz')
            test_target_file_path = os.path.join(transformed_test_dir, 'test_target.npz')

            # logging.info('Saving Transformed Training and Transformed test data')
            # transformed_train_file_path = os.path.join(transformed_dir, transformed_train_dir, 'train.npz')
            # train_target_file_path = os.path.join(transformed_dir, transformed_train_dir, 'train_target.npz')
            # transformed_test_file_path = os.path.join(transformed_dir, transformed_test_dir, 'test.npz')
            # test_target_file_path = os.path.join(transformed_dir, transformed_test_dir, 'test_target.npz')
            
            # os.makedirs(transformed_train_file_path)
            # os.makedirs(train_target_file_path)
            # os.makedirs(transformed_test_file_path)
            # os.makedirs(test_target_file_path)

            save_numpy_array_data(file_path=transformed_train_file_path, array=transformed_train_array)
            save_numpy_array_data(file_path=train_target_file_path, array=train_target_array)
            save_numpy_array_data(file_path=transformed_test_file_path, array=transformed_test_array)
            save_numpy_array_data(file_path=test_target_file_path, array=test_target_array)
            logging.info('Train and test data saved to file')

            # Saving Feature Engineering and preprocessor object

            logging.info('Saving Feature Engineering object') 
#
            # feature_engineering_object_file_path = self.data_transformation_config.feature_engineering_object_file_path
            feature_engineering_object_file_path = self.data_transformation_config.feature_engineering_dir

            save_object(file_path=feature_engineering_object_file_path, obj=fe_obj)

            save_object(file_path=os.path.join(ROOT_DIR, 
                                               PICKLE_FOLDER_KEY_NAME, 
                                               os.path.basename(feature_engineering_object_file_path)), obj=fe_obj)
            
            logging.info('Saving Object') 
#
            # preprocessor_file_path = self.data_transformation_config.preprocessor_file_object_file_path
            preprocessor_file_path = self.data_transformation_config.processor_object_file

            save_object(file_path=preprocessor_file_path, obj=preprocessor)

            save_object(file_path=os.path.join(ROOT_DIR, 
                                               PICKLE_FOLDER_KEY_NAME,
                                               os.path.basename(preprocessor_file_path)), obj=preprocessor)
            
            data_tranforamtion_artifact = DataTransformationArtifact(
                transformed_train_file_path = transformed_train_file_path,
                train_target_file_path = train_target_file_path,
                transformed_test_file_path = transformed_test_file_path,
                test_target_file_path = test_target_file_path, 
                feature_eng_obj_file_path = feature_engineering_object_file_path   
            )

            return data_tranforamtion_artifact


        except Exception as e:
            raise CustomException(e, sys) from e
        

# # line number 233
# # Why do we use fit_transform() method in Class in  when we return preprocessor obj from get_preprocessort() 




# import os 
# import sys
# import pandas as pd
# import numpy as np
# from src.logger import logging
# from src.exception import CustomException
# from src.entity.artifact_entity import *
# from src.entity.config_entity import *
# from src.utils import read_yaml_file,save_object,save_numpy_array_data,create_yaml_file_numerical_columns,create_yaml_file_categorical_columns_from_dataframe
# from src.constant import *
# from sklearn.pipeline import Pipeline
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.pipeline import Pipeline
# import re
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import StandardScaler,MinMaxScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder, FunctionTransformer


# # Reading data in Transformation Schema 
# transformation_yaml = read_yaml_file(file_path=TRANFORMATION_YAML_FILE_PATH)

# # Column data accessed from Schema.yaml
# target_column = transformation_yaml[TARGET_COLUMN_KEY]
# numerical_columns = transformation_yaml[NUMERICAL_COLUMN_KEY] 
# categorical_columns=transformation_yaml[CATEGORICAL_COLUMNS]
# drop_columns=transformation_yaml[DROP_COLUMNS]

# # Transformation 
# outlier_columns=transformation_yaml[OUTLIER_COLUMNS]
# scaling_columns=transformation_yaml[SCALING_COLUMNS]


# class Feature_Engineering(BaseEstimator, TransformerMixin):
    
#     def __init__(self):
        
#         """
#         This class applies necessary Feature Engneering 
#         """
#         logging.info(f"\n{'*'*20} Feature Engneering Started {'*'*20}\n\n")
            
#         logging.info(f" Numerical Columns , Categorical Columns , Target Column initialised in Feature engineering Pipeline ")

#     # Feature Engineering Pipeline 

#                                 ######################### Data Modification ############################
                                
#     def drop_columns(self,X:pd.DataFrame):
#         columns_to_drop=drop_columns
#         logging.info(f"Dropping Columns : {columns_to_drop}")
        
#         X.drop(columns=columns_to_drop,axis=1,inplace=True)
#         return X
#     def replace_spaces_with_underscore(self,df):
#         df = df.rename(columns=lambda x: x.replace(' ', '_'))
#         return df
    
#     def replace_nan_with_random(self,df, column_label):
#         if column_label not in df.columns:
#             print(f"Column '{column_label}' not found in the DataFrame.")
#             return df
        
#         original_data = df[column_label].copy()
#         nan_indices = df[df[column_label].isna()].index
#         num_nan = len(nan_indices)
        
#         existing_values = original_data.dropna().values
#         random_values = np.random.choice(existing_values, num_nan)

#         df.loc[nan_indices, column_label] = random_values
        
#         original_mean = original_data.mean()
#         original_median = original_data.median()
#         new_mean = df[column_label].mean()
#         new_median = df[column_label].median()

#         return df
    
#     def drop_rows_with_nan(self, X: pd.DataFrame, column_label: str):
#         # Log the shape before dropping NaN values
#         logging.info(f"Shape before dropping NaN values: {X.shape}")
        
#         # Drop rows with NaN values in the specified column
#         X = X.dropna(subset=[column_label])
#         #X.to_csv("Nan_values_removed.csv", index=False)
        
#         # Log the shape after dropping NaN values
#         logging.info(f"Shape after dropping NaN values: {X.shape}")
        
#         logging.info("Dropped NaN values.")
#         X = X.reset_index(drop=True)
        
#         return X

#     def trim_outliers_by_quantile(self,df, column_label, lower_quantile=0.05, 
#                                   upper_quantile=0.95):
#         if column_label not in df.columns:
#             print(f"Column '{column_label}' not found in the DataFrame.")
#             return df
        
#         column_data = df[column_label]
        
#         lower_bound = column_data.quantile(lower_quantile)
#         upper_bound = column_data.quantile(upper_quantile)
        
#         trimmed_data = column_data.clip(lower=lower_bound, upper=upper_bound)
        
#         df[column_label] = trimmed_data

#         return df

#     def Removing_outliers(self,X):
        
#         for column in outlier_columns:
#             logging.info(f"Removing Outlier from column :{column}")
#             X=self.trim_outliers_by_quantile(df=X,column_label=column)
            
#         return X
    
#     def run_data_modification(self,data):
    
#         X=data.copy()
        
#         logging.info(" Editing Column Lables ......")
#         X=self.replace_spaces_with_underscore(X)
        
#         try:
#             X = self.drop_columns(X)
#         except Exception as e:
#             print("Test Data does not consists of some Dropped Columns")
        
        
#         logging.info("----------------")
#         logging.info('Replace nan with random Data')
#         for column in ['Artist_Reputation','Height','Width']:
#             # Removing nan rows
#             logging.info(f"Removing NaN values from the column : {column} ")
#             X=self.replace_nan_with_random(X,column_label=column)
        
#         logging.info("----------------")
#         logging.info(' Dropping rows with nan')
#         for column in ['Weight','Material','Remote_Location']:
#             # Removing nan rows
#             logging.info(f"Dropping Rows from column : {column}")
#             X=self.drop_rows_with_nan(X,column_label=column)
            
#         logging.info("----------------")

#         logging.info(" Removing Outliers ")
#         # Removing Outliers
#         X=self.Removing_outliers(X)
        
#         return X
    
#     def data_wrangling(self,X:pd.DataFrame):
#         try: 
#             # Data Modification 
#             data_modified=self.run_data_modification(data=X)
            
#             logging.info(" Data Modification Done")
            
#             return data_modified
    
#         except Exception as e:
#             raise CustomException(e,sys) from e


#     def fit(self,X,y=None):
#         return self
    
#     def transform(self,X:pd.DataFrame,y=None):
#         try:    
#             data_modified=self.data_wrangling(X)
                
#             #data_modified.to_csv("data_modified.csv",index=False)
#             logging.info(" Data Wrangaling Done ")
            
#             logging.info(f"Original Data  : {X.shape}")
#             logging.info(f"Shape Modified Data : {data_modified.shape}")
         
#             return data_modified
#         except Exception as e:
#             raise CustomException(e,sys) from e

# class DataProcessor:
#     def __init__(self, numerical_cols, categorical_cols):
#         self.categorical_cols = categorical_cols
#         self.numerical_cols = numerical_cols

#         # Define preprocessing steps using a Pipeline
#         categorical_transformer = Pipeline(
#             steps=[
#                 ('onehot', OneHotEncoder(handle_unknown='ignore'))
#             ]
#         )
#         numerical_transformer = Pipeline(
#             steps=[
#                 ('log_transform', FunctionTransformer(np.log1p, validate=False))
#             ]
#         )

#         # Create a ColumnTransformer to apply transformations
#         self.preprocessor = ColumnTransformer(
#             transformers=[
#                 ('cat', categorical_transformer, self.categorical_cols),
#                 ('num', numerical_transformer, self.numerical_cols)
#             ],
#             remainder='passthrough'
#         )
        
#     def get_preprocessor(self):
#         return self.preprocessor

#     def fit_transform(self, data):
#         # Fit and transform the data using the preprocessor
#         transformed_data = self.preprocessor.fit_transform(data)
#         return transformed_data
    
    
# class DataTransformation:
#     def __init__(self, data_transformation_config: DataTransformationConfig,
#                     data_ingestion_artifact: DataIngestionArtifact):
#         try:
#             logging.info(f"\n{'*'*20} Data Transformation log started {'*'*20}\n\n")
#             self.data_transformation_config = data_transformation_config
#             self.data_ingestion_artifact = data_ingestion_artifact
                
#         except Exception as e:
#             raise CustomException(e,sys) from e

#     def get_feature_engineering_object(self):
#         try: 
#             feature_engineering = Pipeline(steps = [("fe",Feature_Engineering(  ))])
#             return feature_engineering
#         except Exception as e:
#             raise CustomException(e,sys) from e
#     def separate_numerical_categorical_columns(self,df):
#         numerical_columns = []
#         categorical_columns = []

#         for column in df.columns:
#             if df[column].dtype == 'int64' or df[column].dtype == 'float64':
#                 numerical_columns.append(column)
#             else:
#                 categorical_columns.append(column)

#         return numerical_columns, categorical_columns

#     def initiate_data_transformation(self):
#         try:
#             # Data validation Artifact ------>Accessing train and test files 
#             logging.info(f"Obtaining training and test file path.")
#             train_file_path = self.data_ingestion_artifact.train_file_path
#             test_file_path = self.data_ingestion_artifact.test_file_path
#             logging.info(f"Loading training and test data as pandas dataframe.")
#             logging.info(f" Accessing train file from :{train_file_path}\
#                              Test File Path:{test_file_path} ")      
#             train_df = pd.read_csv(train_file_path)
#             test_df = pd.read_csv(test_file_path)
            
#             logging.info(f"Target Column :{target_column}")
#             logging.info(f"Numerical Column :{numerical_columns}")
#             logging.info(f"Categorical Column :{categorical_columns}")

#             col = numerical_columns + categorical_columns+target_column
#             # All columns 
#             logging.info("All columns: {}".format(col))
            
#             # Feature Engineering 
#             logging.info(f"Obtaining feature engineering object.")
#             fe_obj = self.get_feature_engineering_object()
            
#             logging.info(f"Applying feature engineering object on training dataframe and testing dataframe")
#             logging.info(">>>" * 20 + " Training data " + "<<<" * 20)
#             logging.info(f"Feature Enineering - Train Data ")
#             train_df = fe_obj.fit_transform(X=train_df)
#             logging.info(">>>" * 20 + " Test data " + "<<<" * 20)
#             logging.info(f"Feature Enineering - Test Data ")
#             test_df = fe_obj.transform(X=test_df)
            
#             # Train Data 
#             logging.info(f"Feature Engineering of train and test Completed.")
#             feature_eng_train_df:pd.DataFrame = train_df.copy()
#           #  feature_eng_train_df.to_csv("feature_eng_train_df.csv")
#             logging.info(f" Columns in feature enginering Train {feature_eng_train_df.columns}")
#             logging.info(f"Feature Engineering - Train Completed")
            
#             # Test Data
#             feature_eng_test_df:pd.DataFrame = test_df.copy()
#            # feature_eng_test_df.to_csv("feature_eng_test_df.csv")
#             logging.info(f" Columns in feature enginering test {feature_eng_test_df.columns}")
#             logging.info(f"Saving feature engineered training and testing dataframe.")
            
#             # Getting numerical and categorical of Transformed data 
            
#             # Train and Test 
#             input_feature_train_df = feature_eng_train_df.drop(columns=target_column,axis=1)
#             train_target_array=feature_eng_train_df[target_column]
#             input_feature_test_df= feature_eng_test_df.drop(columns=target_column,axis=1)
#             test_target_array=feature_eng_test_df[target_column]
  
#                                     ############ Input Fatures transformation########
#             ### Preprocessing 
#             logging.info("*" * 20 + " Applying preprocessing object on training dataframe and testing dataframe " + "*" * 20)
            
#             logging.info(f" Scaling Columns : {scaling_columns}")
        
#             # Transforming Data 
#             numerical_cols,categorical_cols=self.separate_numerical_categorical_columns(df=input_feature_train_df)
            
#             # Saving column labels for prediction
#             create_yaml_file_numerical_columns(column_list=numerical_cols,
#                                                yaml_file_path=PREDICTION_YAML_FILE_PATH)
            
#             create_yaml_file_categorical_columns_from_dataframe(dataframe=input_feature_train_df,categorical_columns=categorical_cols,
#                                                                 yaml_file_path=PREDICTION_YAML_FILE_PATH)
            
            
#             logging.info(f" Transformed Data Numerical Columns :{numerical_cols}")
#             logging.info(f" Transformed Data Categorical Columns :{categorical_cols}")
            
#             # Setting the columns order
#             column_order=numerical_cols+categorical_cols
#             input_feature_train_df=input_feature_train_df[column_order]
#             input_feature_test_df=input_feature_test_df[column_order]
 
#             data_preprocessor=DataProcessor(numerical_cols=numerical_cols,categorical_cols=categorical_cols)
            
#             preprocessor=data_preprocessor.get_preprocessor()
        
#             transformed_train_array=data_preprocessor.fit_transform(data=input_feature_train_df)
            
#             train_target_array=train_target_array
            
#             transformed_test_array=data_preprocessor.fit_transform(data=input_feature_test_df)
#             test_target_array=test_target_array
            
#             # Log the shape of Transformed Train
#             logging.info("------- Transformed Data -----------")
            
#             # Adding target column to transformed dataframe
#             transformed_train_dir = self.data_transformation_config.transformed_train_dir
#             transformed_test_dir = self.data_transformation_config.transformed_test_dir    
            
#             os.makedirs(transformed_train_dir,exist_ok=True)
#             os.makedirs(transformed_test_dir,exist_ok=True)

#             ## Saving transformed train and test file
#             logging.info("Saving Transformed Train and Transformed test Data")
#             transformed_train_file_path = os.path.join(transformed_train_dir,"train.npz")
#             train_target_file_path=os.path.join(transformed_train_dir,"train_target.npz")
#             transformed_test_file_path = os.path.join(transformed_test_dir,"test.npz")
#             test_target_file_path=os.path.join(transformed_test_dir,"test_target.npz")
            
#             save_numpy_array_data(file_path = transformed_train_file_path, array = transformed_train_array)
#             save_numpy_array_data(file_path = train_target_file_path, array = train_target_array)
#             save_numpy_array_data(file_path = transformed_test_file_path, array = transformed_test_array)
#             save_numpy_array_data(file_path = test_target_file_path , array = test_target_array)
#             logging.info("Train and Test Data  saved")
           
#                          ###############################################################

#             ### Saving Feature engineering and preprocessor object 
#             logging.info("Saving Feature Engineering Object")
#             feature_engineering_object_file_path = self.data_transformation_config.feature_engineering_object_file_path
#             save_object(file_path = feature_engineering_object_file_path,obj = fe_obj)
#             save_object(file_path=os.path.join(ROOT_DIR,PIKLE_FOLDER_NAME_KEY,
#                                  os.path.basename(feature_engineering_object_file_path)),obj=fe_obj)

#             ### Saving FFeature engineering and preprocessor object 
#             logging.info("Saving  Object")
#             preprocessor_file_path = self.data_transformation_config.preprocessor_file_object_file_path
#             save_object(file_path = preprocessor_file_path,obj = preprocessor)
#             save_object(file_path=os.path.join(ROOT_DIR,PIKLE_FOLDER_NAME_KEY,
#                                  os.path.basename(preprocessor_file_path)),obj=preprocessor)

#             data_transformation_artifact = DataTransformationArtifact(
#                                                                         transformed_train_file_path =transformed_train_file_path,
#                                                                         train_target_file_path=train_target_file_path,
#                                                                         transformed_test_file_path = transformed_test_file_path,
#                                                                         test_target_file_path=test_target_file_path,
#                                                                         feature_engineering_object_file_path = feature_engineering_object_file_path)
            
#             logging.info(f"Data Transformation Artifact: {data_transformation_artifact}")
#             return data_transformation_artifact
#         except Exception as e:
#             raise CustomException(e,sys) from e

#     def __del__(self):
#         logging.info(f"\n{'*'*20} Data Transformation log completed {'*'*20}\n\n")