import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.config import ordinal_features, nominal_features, categorical_features,continuous_numeric_features,fill_none_cols,zero_fill_cols,high_cardinality_features, target_feature    
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
        logging.info("Entered the data ingestion method or component")
        try:
            file_path = os.path.join("notebook", "data", "processed_data", "saleprice_cleaned_dataset.csv")
            df = pd.read_csv(file_path)
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            '''
            self.ingestion_config.raw_data_path → where the file will be saved (e.g., artifacts/data.csv).

            index=False → don’t write DataFrame’s index as a separate column.

            header=True → include column names as the first row.
            '''
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data= obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_array, test_array, _ = data_transformation.initiate_data_transformation(train_data, test_data, ordinal_features, nominal_features,high_cardinality_features, continuous_numeric_features,fill_none_cols,zero_fill_cols, target_feature)

    modeltrainer=ModelTrainer()
    modeltrainer.initiate_model_trainer(train_array, test_array)

