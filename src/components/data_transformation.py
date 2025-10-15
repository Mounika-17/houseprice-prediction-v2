import sys
import os
from dataclasses import dataclass
from src.logger import logger
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder

from src.exception import CustomException

import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# Custom Transformers
# -----------------------------
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import yeojohnson

class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, fill_none_cols=None, zero_fill_cols=None):
        # store the parameters exactly as they are passed. Otherwise, if they are None, we set them to empty lists causes an error when the gridsearchCV tries to clone the transformer.
        self.fill_none_cols = fill_none_cols 
        self.zero_fill_cols = zero_fill_cols

    def fit(self, X, y=None):
        X_ = X.copy()
        # Initialize learned fill_values_ as an empty dictionary
        self.fill_values_ = {}

        # If fill_none_cols or zero_fill_cols are None, set them to empty lists
        fill_none_cols = self.fill_none_cols if self.fill_none_cols is not None else []
        zero_fill_cols = self.zero_fill_cols if self.zero_fill_cols is not None else []
        # Learn rules for all other columns
        for col in X_.columns:
            if col in fill_none_cols or col in zero_fill_cols:  # The purpose of fit is to learn from the training data (compute medians, modes, etc.).But for fill_none_cols and zero_fill_cols, we don’t need to “learn” anything. We just need to transform them so we handled it in transform method.
                continue

            if X_[col].dtype == "object":
                if X_[col].isnull().sum() > 0:
                    self.fill_values_[col] = X_[col].mode()[0]
            else:
                if X_[col].isnull().sum() > 0:
                    skewness = X_[col].skew()
                    if skewness > 1 or skewness < -1:
                        self.fill_values_[col] = X_[col].median()
                    else:
                        self.fill_values_[col] = X_[col].mean()
        return self

    def transform(self, X):
        X_ = X.copy()

        fill_none_cols = self.fill_none_cols if self.fill_none_cols is not None else []
        zero_fill_cols = self.zero_fill_cols if self.zero_fill_cols is not None else []
        # Apply fixed rules
        for col in fill_none_cols:
            if col in X_.columns:
                X_[col] = X_[col].fillna("None")

        for col in zero_fill_cols:
            if col in X_.columns:
                X_[col] = X_[col].fillna(0)

        # Apply learned rules
        for col, fill_value in self.fill_values_.items():
            if col in X_.columns:
                X_[col] = X_[col].fillna(fill_value)

        # Handle unseen missing columns (only in test)
        for col in X_.columns:
            if X_[col].isnull().sum() > 0 and col not in self.fill_values_:
                if X_[col].dtype == "object":
                    fill_value = X_[col].mode()[0]
                else:
                    skewness = X_[col].skew()
                    if skewness > 1 or skewness < -1:
                        fill_value = X_[col].median()
                    else:
                        fill_value = X_[col].mean()
                X_[col] = X_[col].fillna(fill_value)

        return X_
    


class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, discrete_threshold=10, skew_threshold=0.75):
       # self.outlier_columns = outlier_columns
        self.discrete_threshold = discrete_threshold
        self.skew_threshold = skew_threshold
        self.params_ = {}
        self.outlier_columns = []
    # we need to give y=None as the scikit-learn’s Pipeline, GridSearchCV, cross_val_score, and other utilities always pass both X and y to .fit(), even for unsupervised transformers (like imputers, scalers, outlier handlers, etc.) even if they are not used.
    # This is to ensure compatibility with scikit-learn's API.
    def fit(self, X, y=None):
        X_ = X.copy()
        numeric_columns= X_.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if X_[col].dropna().empty:
                continue
            unique_values = X_[col].nunique()
            col_type = 'discrete' if unique_values <= self.discrete_threshold else 'continuous' 
            Q1 = X_[col].quantile(0.25)
            Q3 = X_[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            extreme_outliers = ((X_[col] < lower_bound) | (X_[col] > upper_bound)).sum()
            if extreme_outliers > 0:
                self.outlier_columns.append(col)
                if col_type == 'discrete':
                    self.params_[col] = ('cap', lower_bound, upper_bound)
                else:
                    skewness = X_[col].skew()
                    if abs(skewness) > self.skew_threshold:
                        if (X_[col] >= 0).all():
                            self.params_[col] = ('log',)
                        else:
                            self.params_[col] = ('yeojohnson',)
                    else:
                        self.params_[col] = ('cap', lower_bound, upper_bound)
        return self  # fit method returns the self which is a common practice in scikit-learn to allow for method chaining
    
    def transform(self, X):
        X_ = X.copy()
        for col, params in self.params_.items():
            if params[0] == 'cap':
                _, lower, upper = params
                #X_[col] < lower compares each value in the column to the lower bound. It returns a boolean Series where each entry is True if the corresponding value is less than the lower bound and False otherwise.
                X_[col] = np.where(X_[col] < lower, lower,
                                   np.where(X_[col] > upper, upper, X_[col]))
            elif params[0] == 'log':
                X_[col] = np.log1p(X_[col])
            elif params[0] == 'yeojohnson':
                X_[col], _ = yeojohnson(X_[col])
        return X_ # transform method returns the transformed DataFrame, allowing for further processing or model fitting

# -----------------------------
# Config
# -----------------------------

class DataTransformation:

    def build_preprocessor(self, ordinal_features, nominal_features, high_card_features, continuous_numeric_features):
        preprocessor = ColumnTransformer(
            transformers=[
                ('ord', OrdinalEncoder(categories=[ordinal_features[col] for col in ordinal_features]),
                list(ordinal_features.keys())),
                ('nom', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'),
                nominal_features),
                ('high', TargetEncoder(cols=high_card_features), high_card_features),
                ('num', StandardScaler(), continuous_numeric_features)
            ],
            remainder='drop'
    )
        return preprocessor

    def get_data_transformer_object(self, ordinal_features, nominal_features, high_card_features, continuous_numeric_features,fill_none_cols,
        zero_fill_cols):
        '''Return a preprocessing pipeline (not fitted yet).'''
        try:
            preprocessor = self.build_preprocessor(ordinal_features, nominal_features, high_card_features, continuous_numeric_features)
            custmomimputer= CustomImputer(
                fill_none_cols,
                zero_fill_cols
            )
            pipeline = Pipeline([
                ('custom_imputer', custmomimputer),
                ('outliers', OutlierHandler()),
                ('encode_scale', preprocessor),
            ])
            logger.info("Preprocessing object created")
            return pipeline
        
        except Exception as e:
            raise CustomException(e, sys)
        