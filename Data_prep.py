import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

data = pd.read_csv("Dataset/Hospital_dataset.csv")

# Drop irrelevant columns
data = data.drop(["readmission_risk_score", "admission_date", "patient_id"], axis=1) # No discharge date and we already have season

# Handle missing data
print(data.isna().sum()) # None

# Binary encoding
data["gender"] = data["gender"].map({"Male": 1, "Female": 0})

# Formatting 
# Data is formatted correctly

# Variable deriviations
# None done yet.

# Save the new data into a csv
data.to_csv("Dataset/prepped_hospital_data.csv", index=False)