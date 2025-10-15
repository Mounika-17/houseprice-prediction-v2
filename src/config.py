ordinal_features = {
    'KitchenQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex']
}

nominal_features = [
    'MSZoning',
    'Neighborhood'
]

categorical_features = [
    'MSZoning',
    'Neighborhood',
    'KitchenQual'
]

continuous_numeric_features = [
    'LotFrontage',
    'LotArea',
    'OverallQual',
    'OverallCond',
    'YearBuilt',
    'YearRemodAdd',
    'GrLivArea',
    'FullBath',
    'BedroomAbvGr',
    'GarageCars',
    'GarageArea',
    'Fireplaces',
    'TotalBsmtSF'
]

fill_none_cols = []  # No categorical features with 'None' in this subset
zero_fill_cols = []  # No numeric features need 0 fill here
high_cardinality_features = []
target_feature = 'SalePrice'