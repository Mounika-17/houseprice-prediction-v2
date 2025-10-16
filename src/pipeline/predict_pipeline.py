import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import numpy as np


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        '''
        Predicts the house price with the given input features.
        model.pkl already includes the fitted preprocessor + model.
        '''
        try:
            model_path = "artifacts/model.pkl"
            model = load_object(file_path=model_path)

            # Directly predict since model.pkl has the preprocessing inside
            preds_log = model.predict(features)
            # Convert back from log scale to original scale
            preds = np.expm1(preds_log)  # np.expm1 is used to reverse np.log1p
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    '''
    This class maps user input from a web form to a DataFrame.
    It ensures that the input data matches the features used during model training.
    '''

    def __init__(self,
                 MSZoning: str,
                 Neighborhood: str,
                 LotFrontage: float,
                 LotArea: int,
                 OverallQual: int,
                 OverallCond: int,
                 YearBuilt: int,
                 YearRemodAdd: int,
                 GrLivArea: int,
                 FullBath: int,
                 BedroomAbvGr: int,
                 KitchenQual: str,
                 GarageCars: int,
                 GarageArea: int,
                 Fireplaces: int,
                 TotalBsmtSF: int = 0):
        self.MSZoning = MSZoning
        self.Neighborhood = Neighborhood
        self.LotFrontage = LotFrontage
        self.LotArea = LotArea
        self.OverallQual = OverallQual
        self.OverallCond = OverallCond
        self.YearBuilt = YearBuilt
        self.YearRemodAdd = YearRemodAdd
        self.GrLivArea = GrLivArea
        self.FullBath = FullBath
        self.BedroomAbvGr = BedroomAbvGr
        self.KitchenQual = KitchenQual
        self.GarageCars = GarageCars
        self.GarageArea = GarageArea
        self.Fireplaces = Fireplaces
        self.TotalBsmtSF = TotalBsmtSF

    def get_data_as_data_frame(self):
        '''
        Converts the user inputs into a DataFrame
        '''
        try:
            # self.__dict__ - It returns a dictionary containing all the instance variables and their current values. .copy() - creates a copy of all the instance variables (attributes) of an object and stores it in a dictionary called input_dict.
            input_dict = self.__dict__.copy()
            # Create a DataFrame from the input data.
            df = pd.DataFrame([input_dict])
            return df

        except Exception as e:
            raise CustomException(e, sys)
