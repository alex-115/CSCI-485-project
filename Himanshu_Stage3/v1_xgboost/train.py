"""
Stage 3 - XGBoost run v1 (Himanshu Singh Rao)
=============================================

First XGBoost tuning pass for 30-day readmission prediction.

Search:    RandomizedSearchCV, n_iter=40, cv=5, scoring=average_precision
Calib:     none (raw model probabilities)
Threshold: F1-max on the validation precision-recall curve

Inputs:
    ../../Dataset/prepped_hospital_data.csv

Outputs (next to this file):
    best_params.json
    metrics.csv
    results.txt
    figures/roc.png
    figures/pr.png
    figures/confusion.png
    figures/calibration.png
    figures/top_features.png
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.calibration import calibration_curve
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
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
DATA_PATH = PROJECT_ROOT / "Dataset" / "prepped_hospital_data.csv"
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_ITER = 40

NUMERIC_COLS = [
    "age",
    "comorbidities_count",
    "length_of_stay",
    "medications_count",
    "followup_visits_last_year",
    "prev_readmissions",
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


def main() -> None:
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()
    print(
        f"Splits  train={len(X_train)}  val={len(X_val)}  test={len(X_test)}\n"
        f"Positive rate train={y_train.mean():.3f}  val={y_val.mean():.3f}  test={y_test.mean():.3f}"
    )

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = float(neg / pos)
    print(f"scale_pos_weight (neg/pos on train) = {scale_pos_weight:.4f}")

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
        "xgb__n_estimators":     randint(200, 900),
        "xgb__learning_rate":    loguniform(1e-2, 3e-1),
        "xgb__max_depth":        randint(3, 9),
        "xgb__min_child_weight": randint(1, 10),
        "xgb__subsample":        uniform(0.6, 0.4),
        "xgb__colsample_bytree": uniform(0.6, 0.4),
        "xgb__reg_lambda":       loguniform(1e-2, 1e1),
        "xgb__gamma":            uniform(0.0, 5.0),
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

    best_model = search.best_estimator_

    y_val_probs = best_model.predict_proba(X_val)[:, 1]
    p, r, t = precision_recall_curve(y_val, y_val_probs)
    f1 = 2 * p * r / (p + r + 1e-12)
    f1 = f1[:-1]
    optimal_threshold = float(t[int(np.argmax(f1))])
    print(f"\nOptimal threshold (max F1 on val): {optimal_threshold:.4f}")
    print(f"  Validation ROC-AUC: {roc_auc_score(y_val, y_val_probs):.4f}")
    print(f"  Validation PR-AUC : {average_precision_score(y_val, y_val_probs):.4f}")

    y_test_probs = best_model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_probs >= optimal_threshold).astype(int)
    test_roc = roc_auc_score(y_test, y_test_probs)
    test_pr = average_precision_score(y_test, y_test_probs)
    test_brier = brier_score_loss(y_test, y_test_probs)
    test_f2 = fbeta_score(y_test, y_test_pred, beta=2)
    test_cm = confusion_matrix(y_test, y_test_pred)

    print("\n=== FINAL TEST SET PERFORMANCE (XGBoost v1, tuned) ===")
    print(classification_report(y_test, y_test_pred, digits=3))
    print(f"ROC-AUC: {test_roc:.4f}\nPR-AUC : {test_pr:.4f}\nBrier  : {test_brier:.4f}\nF2     : {test_f2:.4f}")
    print(f"Confusion matrix:\n{test_cm}")
    flagged = float((y_test_probs >= optimal_threshold).mean()) * 100.0
    print(f"Flagged high-risk: {flagged:.2f}%")

    json_safe = {k: (float(v) if hasattr(v, "item") else v) for k, v in best_params.items()}
    with open(HERE / "best_params.json", "w") as f:
        json.dump({
            "version": "v1",
            "search": {"method": "RandomizedSearchCV", "n_iter": N_ITER, "cv": 5,
                       "scoring": "average_precision"},
            "calibration": "none",
            "best_params": json_safe,
            "scale_pos_weight": scale_pos_weight,
            "cv_best_pr_auc": float(search.best_score_),
            "optimal_threshold": optimal_threshold,
        }, f, indent=2)

    pd.DataFrame([{
        "model": "XGBoost v1 (tuned)",
        "test_roc_auc": test_roc, "test_pr_auc": test_pr,
        "test_brier": test_brier, "test_f2": test_f2,
        "threshold": optimal_threshold,
        "tn": int(test_cm[0, 0]), "fp": int(test_cm[0, 1]),
        "fn": int(test_cm[1, 0]), "tp": int(test_cm[1, 1]),
    }]).to_csv(HERE / "metrics.csv", index=False)

    fpr, tpr, _ = roc_curve(y_test, y_test_probs)
    plt.figure(figsize=(6.5, 5.5))
    plt.plot(fpr, tpr, label=f"XGBoost v1 (AUC = {test_roc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="grey", label="Chance")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("XGBoost v1 - Test ROC")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(FIG_DIR / "roc.png", dpi=150); plt.close()

    base_rate = float(np.mean(y_test))
    p, r, _ = precision_recall_curve(y_test, y_test_probs)
    plt.figure(figsize=(6.5, 5.5))
    plt.plot(r, p, label=f"XGBoost v1 (AP = {test_pr:.3f})")
    plt.hlines(base_rate, 0, 1, colors="grey", linestyles="--",
               label=f"Base rate = {base_rate:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("XGBoost v1 - Test Precision-Recall")
    plt.legend(loc="lower left"); plt.tight_layout()
    plt.savefig(FIG_DIR / "pr.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    im = ax.imshow(test_cm, cmap="Blues")
    ax.set_title(f"XGBoost v1 - Test confusion (thr={optimal_threshold:.3f})")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, test_cm[i, j], ha="center", va="center",
                    color="black" if test_cm[i, j] < test_cm.max() / 2 else "white")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion.png", dpi=150); plt.close(fig)

    prob_true, prob_pred = calibration_curve(y_test, y_test_probs, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6.0, 5.5))
    plt.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    plt.plot(prob_pred, prob_true, marker="o", label=f"XGBoost v1 (Brier={test_brier:.3f})")
    plt.xlabel("Predicted probability"); plt.ylabel("Observed positive rate")
    plt.title("XGBoost v1 - Test calibration")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(FIG_DIR / "calibration.png", dpi=150); plt.close()

    booster = best_model.named_steps["xgb"].get_booster()
    pre = best_model.named_steps["preprocess"]
    feature_names = list(pre.get_feature_names_out())
    importance = booster.get_score(importance_type="gain")
    idx_to_name = {f"f{i}": feature_names[i] for i in range(len(feature_names))}
    series = pd.Series(
        {idx_to_name.get(k, k): v for k, v in importance.items()}
    ).sort_values(ascending=True).tail(15)
    plt.figure(figsize=(8, 6))
    series.plot(kind="barh")
    plt.xlabel("Gain"); plt.title("XGBoost v1 - top 15 features (gain)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "top_features.png", dpi=150); plt.close()

    with open(HERE / "results.txt", "w") as f:
        f.write("=== XGBoost v1 (tuned, no calibration) - test-set results ===\n\n")
        f.write(f"Search: RandomizedSearchCV n_iter={N_ITER}, cv=5, scoring=average_precision\n")
        f.write(f"Calibration: none\n\n")
        f.write(f"Best CV PR-AUC: {search.best_score_:.4f}\n")
        f.write(f"Best params: {json_safe}\n")
        f.write(f"scale_pos_weight = {scale_pos_weight:.4f}\n\n")
        f.write(f"Optimal threshold (max F1 on val): {optimal_threshold:.4f}\n\n")
        f.write(f"ROC-AUC: {test_roc:.6f}\n")
        f.write(f"PR-AUC : {test_pr:.6f}\n")
        f.write(f"Brier  : {test_brier:.6f}\n")
        f.write(f"F2     : {test_f2:.6f}\n\n")
        f.write(f"Confusion matrix:\n{test_cm}\n\n")
        f.write("Classification report:\n")
        f.write(classification_report(y_test, y_test_pred, digits=3))
        f.write(f"\nFlagged high-risk: {flagged:.2f}%\n")
        f.write("\nTop 15 features (by gain):\n")
        f.write(series.iloc[::-1].to_string()); f.write("\n")


if __name__ == "__main__":
    main()
