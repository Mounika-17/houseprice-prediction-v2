import pandas as pd
import os

# Path to your existing full dataset
input_path = os.path.join("notebook", "data", "processed_data", "saleprice_cleaned_dataset.csv")

# Read your full dataset
df = pd.read_csv(input_path)

# Select the most important columns from 81 columms
selected_columns = [
    "MSZoning",
    "Neighborhood",
    "LotFrontage",
    "LotArea",
    "OverallQual",
    "OverallCond",
    "YearBuilt",
    "YearRemodAdd",
    "GrLivArea",
    "FullBath",
    "BedroomAbvGr",
    "KitchenQual",
    "GarageCars",
    "GarageArea",
    "Fireplaces",
    "TotalBsmtSF",
    "SalePrice"  # Include target feature
]

# Subset the data
lightweight_df = df[selected_columns]

# Define output path
output_path = os.path.join("notebook", "data", "processed_data", "lightweight_data.csv")

# Save the new dataset
lightweight_df.to_csv(output_path, index=False)

print(f"Lightweight dataset created and saved at: {output_path}")
print(f"Shape: {lightweight_df.shape}")
print(f"Columns: {list(lightweight_df.columns)}")
