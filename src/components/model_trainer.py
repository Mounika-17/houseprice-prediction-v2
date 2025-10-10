import os
import sys
import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, mean_squared_error
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from sklearn.linear_model import (
    LinearRegression, RidgeCV, LassoCV, ElasticNetCV
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Optional imports (only if installed)
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None


# -----------------------------
# Config class
# -----------------------------
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


# -----------------------------
# RMSE Scorer
# -----------------------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_scorer = make_scorer(rmse, greater_is_better=False)


# -----------------------------
# Main Class
# -----------------------------
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            # Define models
            base_models = {
                "LinearRegression": LinearRegression(),
                "RidgeCV": RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5),
                "LassoCV": LassoCV(alphas=[0.001, 0.1, 1.0, 10.0], cv=5, max_iter=50000),
                "ElasticNetCV": ElasticNetCV(
                    alphas=[0.001, 0.1, 1.0, 10.0],
                    l1_ratio=[0.1, 0.5, 0.9],
                    cv=5,
                    max_iter=50000,
                ),
                "DecisionTree": DecisionTreeRegressor(random_state=42),
                "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
                "GradientBoosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
                "AdaBoost": AdaBoostRegressor(n_estimators=200, random_state=42),
                "SVR": SVR(kernel='rbf', C=100, gamma='auto'),
                "KNeighbors": KNeighborsRegressor(n_neighbors=5)
            }

            # Add optional models if available
            if XGBRegressor is not None:
                base_models["XGBRegressor"] = XGBRegressor(
                    n_estimators=300, learning_rate=0.05, random_state=42
                )
            if CatBoostRegressor is not None:
                base_models["CatBoostRegressor"] = CatBoostRegressor(
                    iterations=300, learning_rate=0.05, verbose=False, random_state=42
                )

            # Evaluate each model using K-Fold CV
            results = {}
            cv = KFold(n_splits=5, shuffle=True, random_state=42)

            for name, model in base_models.items():
                logging.info(f"Evaluating {name} with cross-validation...")
                scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=rmse_scorer)
                results[name] = {
                    "CV_RMSE_Mean": -scores.mean(),
                    "CV_RMSE_Std": scores.std(),
                }

            # Select best model
            best_model_name = min(results, key=lambda k: results[k]['CV_RMSE_Mean'])
            best_model = base_models[best_model_name]

            logging.info(f"Best Model: {best_model_name} with RMSE {results[best_model_name]['CV_RMSE_Mean']:.4f}")

            # Retrain best model on full training data
            best_model.fit(X_train, y_train)

            # Evaluate on test data
            y_pred = best_model.predict(X_test)
            test_rmse = rmse(y_test, y_pred)
            logging.info(f"Test RMSE: {test_rmse:.4f}")

            # Save model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            logging.info(f"Saved best model: {best_model_name}")
            return test_rmse, best_model_name, results

        except Exception as e:
            raise CustomException(e, sys)
