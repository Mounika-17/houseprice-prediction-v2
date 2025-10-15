from src.logger import logger
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.config import ordinal_features, nominal_features, categorical_features,continuous_numeric_features,fill_none_cols,zero_fill_cols,high_cardinality_features, target_feature  
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    logger.info("Starting the training pipeline...")
    logger.info("Step 1: Data Ingestion")

    # Step 1: Data Ingestion
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
    logger.info(f"Data ingestion completed. Train data path: {train_data_path}, Test data path: {test_data_path}")
    logger.info("Step 2: Data Transformation")

    # Step 2: Data Transformation (create preprocessor pipeline)
    data_transformation = DataTransformation() 
    preprocessor = data_transformation.get_data_transformer_object(ordinal_features, nominal_features, high_cardinality_features, continuous_numeric_features, fill_none_cols, zero_fill_cols)
    logger.info("Data transformation pipeline created.")
    logger.info("Step 3: Model Training")

    # Step 3: Model Training
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(train_data_path, test_data_path, preprocessor,target_feature)
