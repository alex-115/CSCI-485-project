import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, precision_recall_curve, accuracy_score, precision_score, f1_score, recall_score
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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

#Training begins here

#Important parameters to tune
# regularization hyperparameter C: controls how strongly the model penalizes large coefficients
# The smaller the strong the penalization, simpler model. 

# Penalty type: Ridge, Lasso, and Elastic net.
# Liblinear (good for small datasets, supports l1), lbfs (fast, supports l2), saga(supports l1, elsatic net, large datasets)

# for imablanced datasets i will use class_weight='balanced'

param_grid = {
    "logistic__C": [0.001, 0.01, 0.1, 1, 10, 100],
    "logistic__l1_ratio": [0, 0.25, 0.5, 0.75, 1],   # 0 = L2, 1 = L1
    "logistic__solver": ["saga"],   # saga is the only solver supporting l1_ratio
    "logistic__class_weight": [None, "balanced"]
}

pipeline = Pipeline(steps=[
    ("preprocess", preprocess),
    ("logistic", LogisticRegression(max_iter=2000))
])

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="average_precision", # Class label is highly imbalanced
    cv=5,
    n_jobs = -1,
    verbose=0
)

grid.fit(X_train, y_train)

# Evaluation on Validation Set
best_model = grid.best_estimator_
y_val_probs = best_model.predict_proba(X_val)[:, 1]

print("Best params:", grid.best_params_)

# Calculate Precision-Recall Curve
precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_probs)

# Calculate F1 for every threshold
f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
f1_scores = f1_scores[:-1]   # drop the extra point (precision_recall_curve returns n+1)
best_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[best_idx] 

print(f"Optimal Threshold found on Validation: {optimal_threshold:.4f}")
print(f"Best F1-Score at this threshold: {f1_scores[best_idx]:.4f}")

# Get probabilities for TEST set
y_test_probs = best_model.predict_proba(X_test)[:, 1]

# Apply the threshold found in Step B
y_test_pred_final = (y_test_probs >= optimal_threshold).astype(int)

# Final Metrics
print("--- FINAL TEST SET PERFORMANCE ---")
print(classification_report(y_test, y_test_pred_final))
print("ROC-AUC:", roc_auc_score(y_test, y_test_probs))
print("PR-AUC:", average_precision_score(y_test, y_test_probs))
print(f"Threshold used: {optimal_threshold:.4f}")

# This model is overall good at identifying patients who will be readmitted within 30 days than to those who will not be readmitted
# It is better to have false positives than false negatives.