"""
Stage 3 - XGBoost run v2 (Himanshu Singh Rao)
=============================================

"Can we tune further without overfitting?" follow-up to v1.

Differences from v1 (../v1_xgboost/train.py):
  * RandomizedSearchCV n_iter raised 40 -> 100.
  * Search space adds three knobs:
      - reg_alpha          (L1 on leaves)
      - colsample_bylevel  (extra per-level decorrelation)
      - max_delta_step     (more stable logistic updates under imbalance)
  * learning_rate range widened down to 0.005 (slower learning).
  * n_estimators range widened up to 1500.
  * After tuning, the best estimator is wrapped in CalibratedClassifierCV
    (isotonic, cv=5). Brier is the leading indicator of over-tuning,
    so this is the safety net.
  * Threshold is re-calibrated on validation after isotonic.

Inputs:
    ../../Dataset/prepped_hospital_data.csv

Outputs (next to this file):
    best_params.json
    metrics.csv
    results.txt
    figures/roc.png
    figures/pr.png
    figures/calibration.png
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
import shap

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "Dataset", "prepped_hospital_data.csv")
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_ITER = 100

NUMERIC_COLS = [
    "age", "comorbidities_count", "length_of_stay",
    "medications_count", "followup_visits_last_year", "prev_readmissions",
]
CATEGORICAL_COLS = [
    "season", "region", "primary_diagnosis", "treatment_type",
    "insurance_type", "discharge_disposition",
]


def load_splits():
    df = pd.read_csv(DATA_PATH)
    X = df.drop("label", axis=1)
    y = df["label"].astype(int)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ("num", StandardScaler(), NUMERIC_COLS),
    ], remainder="passthrough")


def best_threshold(y_true, probs) -> float:
    p, r, t = precision_recall_curve(y_true, probs)
    f1 = 2 * p * r / (p + r + 1e-12)
    return float(t[int(np.argmax(f1[:-1]))])


def main() -> None:
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()
    print(f"Splits  train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = float(neg / pos)
    print(f"scale_pos_weight (neg/pos) = {scale_pos_weight:.4f}")

    pipeline = Pipeline(steps=[
        ("preprocess", build_preprocessor()),
        ("xgb", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
        )),
    ])

    param_dist = {
        "xgb__n_estimators":      randint(300, 1500),
        "xgb__learning_rate":     loguniform(5e-3, 2e-1),
        "xgb__max_depth":         randint(3, 9),
        "xgb__min_child_weight":  randint(1, 12),
        "xgb__subsample":         uniform(0.6, 0.4),
        "xgb__colsample_bytree":  uniform(0.6, 0.4),
        "xgb__colsample_bylevel": uniform(0.6, 0.4),
        "xgb__reg_lambda":        loguniform(1e-2, 1e1),
        "xgb__reg_alpha":         loguniform(1e-3, 1e0),
        "xgb__gamma":             uniform(0.0, 5.0),
        "xgb__max_delta_step":    randint(0, 6),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=N_ITER,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        refit=True,
    )

    print(f"\nRunning RandomizedSearchCV  n_iter={N_ITER}, cv=5, scoring=average_precision ...")
    search.fit(X_train, y_train)

    best_params = {k.replace("xgb__", ""): v for k, v in search.best_params_.items()}
    print(f"\nBest CV PR-AUC: {search.best_score_:.4f}")
    print(f"Best params: {best_params}")

    tuned_model = search.best_estimator_

    # SHAP ANALYSIS

    print("\nRunning SHAP analysis on tuned XGBoost model...")

    #   Extract the trained XGB model
    xgb_model = tuned_model.named_steps["xgb"]

    # Extract the preprocessor and transform the test set
    preprocessor = tuned_model.named_steps["preprocess"]
    X_test_t = preprocessor.transform(X_test)

    # Build SHAP explainer
    explainer = shap.TreeExplainer(xgb_model)

    # Compute SHAP values for the test set
    shap_values = explainer.shap_values(X_test_t)
    print("shap values: \n")
    feature_names = tuned_model.named_steps["preprocess"].get_feature_names_out()
    # feature importance
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    global_importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)

    print("\nTop 20 global SHAP features:")
    print(global_importance.head(20))

    #   Summary plot (beeswarm)
    shap.summary_plot(shap_values, X_test_t, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_beeswarm.png", dpi=300)
    plt.close()

    # Bar plot of mean
    shap.summary_plot(shap_values, X_test_t, feature_names=feature_names, show=False, plot_type="bar")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_mean_plot.png", dpi=300)
    plt.close()



    
    



if __name__ == "__main__":
    main()
