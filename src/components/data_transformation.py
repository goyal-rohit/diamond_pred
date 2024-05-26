import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import sys, os, pathlib

from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    out_path = pathlib.Path(__file__).parent.parent.parent.as_posix() + '/artifacts'
    

class DataTransformation:
    def __init__(self) -> None:
        self.data_transofrmation_config = DataTransformationConfig()

    def get_data_transformation_object(self,df):
        logging.info('Data Transformation Method starts')
        try:
            #cat & num columns

            cut_categories = ['Fair','Good', 'Very Good', 'Premium','Ideal']
            clarity_categories = [ 'I1','SI2','SI1','VVS2','VS1','VS2','VVS1','IF']
            color_categories = ['D','E','F','G','H','I','J']

            logging.info('Data Transformation pipeline initiated')

            numerical_cols = df.select_dtypes(exclude='object').columns
            categorical_cols = df.select_dtypes(include='object').columns
            
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scalar',StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder', OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                    ('scalar',StandardScaler())

                ]
            )
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline', cat_pipeline, categorical_cols)
                ]
            )

            logging.info('Data Transformation completed')
            return preprocessor

        except Exception as e:
            logging.info('Exception occured at Data Transformation')
            raise CustomException
        
    def initiate_data_transformation(self,train_data_path,test_data_path):
        
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            target_column = 'price'
            drop_column = ['id',target_column]

            input_feature_train_df = train_df.drop(drop_column, axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(drop_column, axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info('Reading train & Test completed')

            logging.info('Obtaining preprocessing obj')

            preprocessing_obj = self.get_data_transformation_object(input_feature_train_df)

            #Data Transformation
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            logging.info('Preprocessing done')

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_object(
                file_path = self.data_transofrmation_config.out_path,
                file_name = 'preprocessor.pkl',
                obj=preprocessing_obj
            )
            logging.info('Processsor pickle is created and saved')

            return (
                train_arr,
                test_arr
            )
            
        except Exception as e:

            raise CustomException(e,sys)