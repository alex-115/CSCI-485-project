import numpy as np
import pandas as pd

data = pd.read_csv("Hospital_dataset.csv")

# Handle missing data
print(data.isna().sum())

# Type conversions
print(data.info())

data['admission_date'] = pd.to_datetime(data['admission_date'])
print(data['admission_date'].dtype)

# Formatting 

# Variable deriviations

# Standardization

# Save the new data into a csv