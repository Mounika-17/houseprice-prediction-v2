# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = FastAPI(
    title="🏠 Advanced House Price Prediction API",
    version="2.0",
    description="A FastAPI-based deployment of the Advanced House Price Prediction Model"
)

# Define expected input schema using Pydantic
class HouseFeatures(BaseModel):
    MSZoning: str
    Neighborhood: str
    LotFrontage: float
    LotArea: int
    OverallQual: int
    OverallCond: int
    YearBuilt: int
    YearRemodAdd: int
    GrLivArea: int
    FullBath: int
    BedroomAbvGr: int
    KitchenQual: str
    GarageCars: int
    GarageArea: int
    Fireplaces: int
    TotalBsmtSF: int


@app.get("/")
def root():
    return {"message": "House Price Prediction API is live!"}


@app.post("/predict")
def predict_price(data: HouseFeatures):
    try:
        # Convert input JSON to CustomData object
        custom_data = CustomData(
            MSZoning=data.MSZoning,
            Neighborhood=data.Neighborhood,
            LotFrontage=data.LotFrontage,
            LotArea=data.LotArea,
            OverallQual=data.OverallQual,
            OverallCond=data.OverallCond,
            YearBuilt=data.YearBuilt,
            YearRemodAdd=data.YearRemodAdd,
            GrLivArea=data.GrLivArea,
            FullBath=data.FullBath,
            BedroomAbvGr=data.BedroomAbvGr,
            KitchenQual=data.KitchenQual,
            GarageCars=data.GarageCars,
            GarageArea=data.GarageArea,
            Fireplaces=data.Fireplaces,
            TotalBsmtSF=data.TotalBsmtSF
        )

        # Convert to DataFrame for prediction
        pred_df = custom_data.get_data_as_data_frame()

        # Run prediction
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(pred_df)

        return {"Predicted_House_Price": round(float(prediction[0]), 2)}

    except Exception as e:
        return {"error": str(e)}
