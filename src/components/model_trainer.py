import os
import sys
import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor
)
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
)
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.components.data_transformation import DataTransformation
from src.config import ordinal_features, nominal_features, high_cardinality_features, continuous_numeric_features, fill_none_cols, zero_fill_cols

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


# -----------------------------
# Custom RMSE scorer
# -----------------------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


rmse_scorer = make_scorer(rmse, greater_is_better=False)


# -----------------------------
# Model Trainer Class
# -----------------------------
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_data_path, test_data_path, preprocessor):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_data_path[:, :-1],
                train_data_path[:, -1],
                test_data_path[:, :-1],
                test_data_path[:, -1],
            )

            # -----------------------------
            # Get Preprocessor
            # ----------------------------

            cv = KFold(n_splits=5, shuffle=True, random_state=42)

            # -----------------------------
            # Models and their Param Grids
            # -----------------------------
            models_and_params = {
                "LinearRegression": (LinearRegression(), {}),
                "Ridge": (
                    Ridge(), 
                    {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
                    ),
                "Lasso": (
                    Lasso(max_iter=50000), 
                    {"model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0]}
                    ),
                "ElasticNet": (
                    ElasticNet(max_iter=50000),
                    {"model__alpha": [0.001, 0.01, 0.1, 1.0], "model__l1_ratio": [0.1, 0.5, 0.9]}
                ),
                "BayesianRidge": (
                    BayesianRidge(),
                    {"model__alpha_1": [1e-6, 1e-5, 1e-4], "model__alpha_2": [1e-6, 1e-5, 1e-4]}
                ),
                "HuberRegressor": (
                    HuberRegressor(max_iter=50000),
                    {"model__epsilon": [1.1, 1.3, 1.5]}
                ),
                "SVR": (
                    SVR(),
                    {"model__C": [0.1, 1, 10], "model__kernel": ["linear", "rbf"], "model__gamma": ["scale", "auto"]}
                ),
                "RandomForest": (
                    RandomForestRegressor(random_state=42),
                    {"model__n_estimators": [50, 100, 200], "model__max_depth": [5, 10, None],
                     "model__min_samples_split": [2, 5, 10]}
                ),
                "GradientBoosting": (
                    GradientBoostingRegressor(random_state=42),
                    {"model__n_estimators": [50, 100, 200], "model__learning_rate": [0.01, 0.1, 0.2],
                     "model__max_depth": [3, 5]}
                ),
                "AdaBoost": (
                    AdaBoostRegressor(random_state=42),
                    {"model__n_estimators": [50, 100, 200], "model__learning_rate": [0.01, 0.1, 0.5, 1.0]}
                ),
                "XGBoost": (
                    XGBRegressor(eval_metric="rmse", random_state=42),
                    {"model__n_estimators": [100, 200], "model__learning_rate": [0.01, 0.1, 0.2],
                     "model__max_depth": [3, 5, 7]}
                ),
            }

            # -----------------------------
            # Run GridSearchCV for all models
            # -----------------------------
            best_overall_model = None
            # Initializes the best_overall_rmse to positive infinity, ensuring that any model's RMSE will be lower initially.
            # This variable is used to keep track of the lowest RMSE encountered during the model evaluation
            best_overall_rmse = float("inf")
            best_results = {}

            for model_name, (model, param_grid) in models_and_params.items():
                logging.info(f"Running GridSearchCV for {model_name}")

                pipeline = Pipeline(steps=[
                    ("preprocessor", preprocessor),
                    ("model", model)
                ])

                grid_search = GridSearchCV(
                    estimator=pipeline,
                    param_grid=param_grid,
                    cv=cv,
                    scoring=rmse_scorer,
                    n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_

                # Cross-validation scores for the best model
                cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring=rmse_scorer, n_jobs=-1)
                cv_rmse_mean = -cv_scores.mean()
                cv_rmse_std = cv_scores.std()

                # Evaluate on test data
                y_pred_test = best_model.predict(X_test)
                test_rmse = rmse(y_test, y_pred_test)
                test_r2 = r2_score(y_test, y_pred_test)

                logging.info(
                    f"{model_name}: CV RMSE={cv_rmse_mean:.4f} ± {cv_rmse_std:.4f}, "
                    f"Test RMSE={test_rmse:.4f}, R²={test_r2:.4f}"
                )

                results = {
                    "best_params": grid_search.best_params_,
                    "cv_rmse_mean": cv_rmse_mean,
                    "cv_rmse_std": cv_rmse_std,
                    "test_rmse": test_rmse,
                    "test_r2": test_r2
                }

                # Update best overall model
                if cv_rmse_mean < best_overall_rmse:
                    best_overall_model = best_model
                    best_overall_rmse = cv_rmse_mean
                    best_results = {**results, "model": model_name}

            # -----------------------------
            # Save the best model
            # -----------------------------
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_overall_model
            )

            logging.info(f"Best Model: {best_results['model']}")
            logging.info(f"Best Params: {best_results['best_params']}")
            logging.info(f"CV RMSE: {best_results['cv_rmse_mean']:.4f}")
            logging.info(f"Test R²: {best_results['test_r2']:.4f}")

            return best_results

        except Exception as e:
            raise CustomException(e, sys)
