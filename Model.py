import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression  # or any model

data = pd.read_csv("Dataset/prepped_hospital_data.csv")

# Methods for training/validation/testing
# k-fold cross-validation else typical split train: 60-70% -> validation: 15-20% -> test: 15-20%
X = data.drop("label", axis = 1)
y = data["label"]
kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = LogisticRegression(max_iter=1000)
scores = cross_val_score(model, X, y, cv=kf, scoring="accuracy")

# For medium 5k - 100k rows training: 80 -> validation: 10 -> test: 10
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True) # I used shuffle to randomize the selection of data

#Training begins here