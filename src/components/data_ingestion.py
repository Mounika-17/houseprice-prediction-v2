import os
import sys
from src.exception import CustomException
from src.logger import logger
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation  
from src.components.model_trainer import ModelTrainer

import warnings
warnings.filterwarnings("ignore")

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logger.info("Entered the data ingestion method or component")
        try:
            file_path = os.path.join("notebook", "data", "processed_data", "lightweight_data.csv")
            df = pd.read_csv(file_path)
            logger.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            '''
            self.ingestion_config.raw_data_path → where the file will be saved (e.g., artifacts/data.csv).

            index=False → don’t write DataFrame’s index as a separate column.

            header=True → include column names as the first row.
            '''
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logger.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logger.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)

