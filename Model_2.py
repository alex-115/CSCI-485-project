import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression  # or any model
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

data = pd.read_csv("Dataset/prepped_hospital_data.csv")

# Data preprocessing
numeric_cols = [
    "age",
    "comorbidities_count",
    "length_of_stay",
    "medications_count",
    "followup_visits_last_year",
    "prev_readmissions"
]

categorical_cols = [
    "season",
    "region",
    "primary_diagnosis",
    "treatment_type",
    "insurance_type",
    "discharge_disposition"
]
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols)
    ], remainder="passthrough"
)

# Data split 70% training, 15% validation, 15% test
X = data.drop("label", axis=1)
y = data["label"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# Fit preprocessor on training data and transform all sets
X_train_transformed = preprocess.fit_transform(X_train)
X_val_transformed = preprocess.transform(X_val)
X_test_transformed = preprocess.transform(X_test)

#Training begins here
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train_transformed, y_train)

y_val_pred = model.predict(X_val_transformed)
y_val_probs = model.predict_proba(X_val_transformed)[:, 1]

roc_auc = roc_auc_score(y_val, y_val_probs)
pr_auc = average_precision_score(y_val, y_val_probs)

print("=== Random Forest Validation Results ===")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC:  {pr_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred))