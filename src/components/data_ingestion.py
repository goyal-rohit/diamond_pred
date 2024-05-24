import pathlib
import sys
from src.logger import logging
from src.exception import CustomException

import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass


# initialize the data ingestion configuration

@dataclass
class DataIngestionConfig:
    train_data_path: str = None
    test_data_path: str = None


# create a data ingestion class
@dataclass()
class DataIngestion:
    def __init__(self):
        self.out_path = None
        self.raw_data = None
        self.home_dir = None
        self.curr_dir = None
        self.ingestion_config = DataIngestionConfig()
        self.main()

    def main(self):
        try:
            self.curr_dir = pathlib.Path(__file__)
            self.home_dir = self.curr_dir.parent.parent.parent
            # e:\CampusX\projects\PW\diamond_price_pred
            self.raw_data = self.home_dir.as_posix() + '/notebooks/data/train.csv'

            self.out_path = self.home_dir.as_posix() + '/artifacts'

            self.ingestion_config.train_data_path = self.out_path + '/train.csv'
            self.ingestion_config.test_data_path = self.out_path + '/test.csv'

            self.initiate_data_ingestion()

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            logging.info('Exception occurred at main Stage')
            raise CustomException(e, sys)

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion Method starts')

        try:
            df = pd.read_csv(self.raw_data)
            logging.info('Dataset Read as Pandas df')

            pathlib.Path(self.out_path).mkdir(parents=True, exist_ok=True)

            logging.info('Performing Train Test Split')

            train_set, test_set = train_test_split(df, test_size=0.3, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info('Ingestion of data is completed')

        except Exception as e:
            logging.info('Exception occurred at Data Ingestion Stage')
            raise CustomException(e, sys)
