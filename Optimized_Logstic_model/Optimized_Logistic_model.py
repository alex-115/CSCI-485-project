import numpy as np
import pandas as pd
import sys
import os
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.pipeline import Pipeline
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, brier_score_loss, precision_recall_curve, accuracy_score, precision_score, f1_score, recall_score, classification_report, fbeta_score
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

data = pd.read_csv("../Dataset/prepped_hospital_data.csv")

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

# for imablanced datasets i will use class_weight='balanced'

param_grid = {
    "logistic__C": [0.001, 0.01, 0.1, 1, 10, 100],
    "logistic__l1_ratio": [0, 0.25, 0.5, 0.75, 1],   # 0 = L2, 1 = L1
    "logistic__solver": ["saga"],   # saga is the only solver supporting l1_ratio
    "logistic__class_weight": [None, "balanced"]
}

pipeline = Pipeline(steps=[
    ("preprocess", preprocess),
    ("logistic", LogisticRegression(max_iter=3000))
])

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="average_precision", # Class label is highly imbalanced so this is the optimal scoring metric
    cv=5,
    n_jobs = -1,
    verbose=0
)

# Train on training set
grid.fit(X_train, y_train)


# Capacity‑based threshold tuning (top‑K) 
best_model = grid.best_estimator_
capacity = 0.60  # flag top 60%
# Get validation probabilities
y_val_probs = best_model.predict_proba(X_val)[:, 1]
# Compute threshold at the (1 - capacity) percentile
optimal_threshold = np.percentile(y_val_probs, 100 * (1 - capacity))


# Evaluation on test set 
# Get probabilities for test set
y_test_probs = best_model.predict_proba(X_test)[:, 1]

# Apply the threshold
y_test_pred_final = (y_test_probs >= optimal_threshold).astype(int)

feature_names = best_model.named_steps["preprocess"].get_feature_names_out()
coefs = best_model.named_steps["logistic"].coef_.flatten()
coef_table = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefs,
    "odds_ratio": np.exp(coefs)
}).sort_values("coefficient", key=abs, ascending=False)

# Final Metrics : Brier score, F1-score, F2-score, Accuracy, Confusion Matrix, roc-auc
with open("Logstic_Results.txt", "w") as f:
    original_stdout = sys.stdout
    sys.stdout = f
    
    print("=== FINAL TEST SET PERFORMANCE ===")
    print("\n")
    print("Top 3 coeffcients with the strongest impact: ", coef_table.head(3)) # show which factors (like prev_readmissions) are driving the risk score.
    print("\n")

    # Threshold-independent metrics
    print("ROC-AUC:", roc_auc_score(y_test, y_test_probs))
    print("PR-AUC:", average_precision_score(y_test, y_test_probs))
    print("\n")

    # Probability-quality metric
    print("Brier Score:", brier_score_loss(y_test, y_test_probs))
    print("\n")

    # Threshold-dependent metrics
    print("F2-score:", fbeta_score(y_test, y_test_pred_final, beta=1.5))
    print("Accuracy:", accuracy_score(y_test, y_test_pred_final))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred_final))
    print("\n")

    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred_final))
    print("\n")

    # Percentage flagged
    flag_rate = y_test_pred_final.mean()
    print(f"Percentage of patients flagged as high-risk: {flag_rate:.2%}")

    sys.stdout = original_stdout

# 1. Compute the calibration curve
prob_true, prob_pred = calibration_curve(y_test, y_test_probs, n_bins=10)

# 2. Plotting
plt.figure(figsize=(8, 6))
plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Logistic Regression')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')

plt.xlabel('Predicted Probability')
plt.ylabel('Actual Proportion of Positives')
plt.title('Calibration Curve (Reliability Diagram)')
plt.legend()
plt.savefig("Calibration_Curve.png")