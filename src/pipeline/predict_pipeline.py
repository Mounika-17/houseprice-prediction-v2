import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        """
        This function loads the trained model (which already includes preprocessing)
        and predicts house prices for given input features.
        """
        try:
            model_path = "artifacts/model.pkl"
            model = load_object(file_path=model_path)

            # Since model.pkl already includes preprocessor, we can directly predict
            preds = model.predict(features)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    This class maps user input values to a pandas DataFrame
    so that it can be passed to the model for prediction.
    """

    def __init__(self,
                 MSSubClass: int,
                 MSZoning: str,
                 LotFrontage: float,
                 LotArea: int,
                 Street: str,
                 Alley: str,
                 LotShape: str,
                 LandContour: str,
                 Utilities: str,
                 LotConfig: str,
                 LandSlope: str,
                 Neighborhood: str,
                 Condition1: str,
                 Condition2: str,
                 BldgType: str,
                 HouseStyle: str,
                 OverallQual: int,
                 OverallCond: int,
                 YearBuilt: int,
                 YearRemodAdd: int,
                 RoofStyle: str,
                 RoofMatl: str,
                 Exterior1st: str,
                 Exterior2nd: str,
                 MasVnrType: str,
                 MasVnrArea: float,
                 ExterQual: str,
                 ExterCond: str,
                 Foundation: str,
                 BsmtQual: str,
                 BsmtCond: str,
                 BsmtExposure: str,
                 BsmtFinType1: str,
                 BsmtFinSF1: int,
                 BsmtFinType2: str,
                 BsmtFinSF2: int,
                 BsmtUnfSF: int,
                 TotalBsmtSF: int,
                 Heating: str,
                 HeatingQC: str,
                 CentralAir: str,
                 Electrical: str,
                 FirstFlrSF: int,
                 SecondFlrSF: int,
                 LowQualFinSF: int,
                 GrLivArea: int,
                 BsmtFullBath: int,
                 BsmtHalfBath: int,
                 FullBath: int,
                 HalfBath: int,
                 BedroomAbvGr: int,
                 KitchenAbvGr: int,
                 KitchenQual: str,
                 TotRmsAbvGrd: int,
                 Functional: str,
                 Fireplaces: int,
                 FireplaceQu: str,
                 GarageType: str,
                 GarageYrBlt: float,
                 GarageFinish: str,
                 GarageCars: int,
                 GarageArea: int,
                 GarageQual: str,
                 GarageCond: str,
                 PavedDrive: str,
                 WoodDeckSF: int,
                 OpenPorchSF: int,
                 EnclosedPorch: int,
                 ThreeSsnPorch: int,
                 ScreenPorch: int,
                 PoolArea: int,
                 PoolQC: str,
                 Fence: str,
                 MiscFeature: str,
                 MiscVal: int,
                 MoSold: int,
                 YrSold: int,
                 SaleType: str,
                 SaleCondition: str):
        
        # Assign all inputs to object attributes
        self.MSSubClass = MSSubClass
        self.MSZoning = MSZoning
        self.LotFrontage = LotFrontage
        self.LotArea = LotArea
        self.Street = Street
        self.Alley = Alley
        self.LotShape = LotShape
        self.LandContour = LandContour
        self.Utilities = Utilities
        self.LotConfig = LotConfig
        self.LandSlope = LandSlope
        self.Neighborhood = Neighborhood
        self.Condition1 = Condition1
        self.Condition2 = Condition2
        self.BldgType = BldgType
        self.HouseStyle = HouseStyle
        self.OverallQual = OverallQual
        self.OverallCond = OverallCond
        self.YearBuilt = YearBuilt
        self.YearRemodAdd = YearRemodAdd
        self.RoofStyle = RoofStyle
        self.RoofMatl = RoofMatl
        self.Exterior1st = Exterior1st
        self.Exterior2nd = Exterior2nd
        self.MasVnrType = MasVnrType
        self.MasVnrArea = MasVnrArea
        self.ExterQual = ExterQual
        self.ExterCond = ExterCond
        self.Foundation = Foundation
        self.BsmtQual = BsmtQual
        self.BsmtCond = BsmtCond
        self.BsmtExposure = BsmtExposure
        self.BsmtFinType1 = BsmtFinType1
        self.BsmtFinSF1 = BsmtFinSF1
        self.BsmtFinType2 = BsmtFinType2
        self.BsmtFinSF2 = BsmtFinSF2
        self.BsmtUnfSF = BsmtUnfSF
        self.TotalBsmtSF = TotalBsmtSF
        self.Heating = Heating
        self.HeatingQC = HeatingQC
        self.CentralAir = CentralAir
        self.Electrical = Electrical
        self.FirstFlrSF = FirstFlrSF
        self.SecondFlrSF = SecondFlrSF
        self.LowQualFinSF = LowQualFinSF
        self.GrLivArea = GrLivArea
        self.BsmtFullBath = BsmtFullBath
        self.BsmtHalfBath = BsmtHalfBath
        self.FullBath = FullBath
        self.HalfBath = HalfBath
        self.BedroomAbvGr = BedroomAbvGr
        self.KitchenAbvGr = KitchenAbvGr
        self.KitchenQual = KitchenQual
        self.TotRmsAbvGrd = TotRmsAbvGrd
        self.Functional = Functional
        self.Fireplaces = Fireplaces
        self.FireplaceQu = FireplaceQu
        self.GarageType = GarageType
        self.GarageYrBlt = GarageYrBlt
        self.GarageFinish = GarageFinish
        self.GarageCars = GarageCars
        self.GarageArea = GarageArea
        self.GarageQual = GarageQual
        self.GarageCond = GarageCond
        self.PavedDrive = PavedDrive
        self.WoodDeckSF = WoodDeckSF
        self.OpenPorchSF = OpenPorchSF
        self.EnclosedPorch = EnclosedPorch
        self.ThreeSsnPorch = ThreeSsnPorch
        self.ScreenPorch = ScreenPorch
        self.PoolArea = PoolArea
        self.PoolQC = PoolQC
        self.Fence = Fence
        self.MiscFeature = MiscFeature
        self.MiscVal = MiscVal
        self.MoSold = MoSold
        self.YrSold = YrSold
        self.SaleType = SaleType
        self.SaleCondition = SaleCondition

    def get_data_as_data_frame(self):
        """
        Converts input values into a single-row DataFrame.
        """
        try:
            data_dict = self.__dict__
            return pd.DataFrame([data_dict])

        except Exception as e:
            raise CustomException(e, sys)
