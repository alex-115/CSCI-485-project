import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression  # or any model

data = pd.read_csv("Prepped_hospital_data.csv")

# Handle missing data
print(data.isna().sum())

# Type conversions
print(data.info())

data['admission_date'] = pd.to_datetime(data['admission_date'])
print(data['admission_date'].dtype)

# Formatting 
# Data is formatted correctly

# Variable deriviations
# None yet, once ramani comes back with findings then I will implement it into our dataset

# Standardization
#Standardizing predictors is important for models that rely on distances, 
#gradients, or regularization, because differences in feature scale can distort model behavior. 
# I can't standardize any data yet until we choose a model, but here is some code just in case

#scaler = StandardScaler()

#numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns

#scaled_arr = scaler.fit_transform(data[numeric_cols])

#scaled_df = pd.DataFrame(scaled_arr, columns=numeric_cols)

#print("Summary of scaled df")
#scaled_df.info()

# Save the new data into a csv
data.to_csv("prepped_hospital_data.csv", index=False)

# Methods for training/validation/testing
# k-fold cross-validation else typical split train: 60-70% -> validation: 15-20% -> test: 15-20%
# X = data.drop("readmission_risk_score", axis = 1)
# y = data["readmission_risk_score"]
#kf = KFold(n_splits=5, shuffle=True, random_state=42)
#model = LogisticRegression(max_iter=1000)
#scores = cross_val_score(model, X, y, cv=kf, scoring="accuracy")

# For medium 5k - 100k rows training: 80 -> validation: 10 -> test: 10
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True) # I used shuffle to randomize the selection of data