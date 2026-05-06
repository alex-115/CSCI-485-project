"""
Stage 3 - Final 3-way comparison (Himanshu Singh Rao)
=====================================================

Compares the three optimized models on the held-out test split:

  * Logistic Regression (tuned)  - re-tuned inline mirroring
                                   ../../Optimized_Logstic_model/Optimized_Logistic_model.py
  * XGBoost v1 (tuned)           - loads ../v1_xgboost/best_params.json and refits
  * XGBoost v2 (tuned + isotonic) - loads ../v2_xgboost/best_params.json,
                                    refits, then re-applies isotonic calibration

Inputs:
    ../../Dataset/prepped_hospital_data.csv
    ../v1_xgboost/best_params.json
    ../v2_xgboost/best_params.json

Outputs (next to this file):
    metrics.csv          # 3-way table of test metrics
    results.txt          # full text dump including v1 vs v2 head-to-head
    figures/roc.png
    figures/pr.png
    figures/confusion_grid.png
    figures/calibration.png
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
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
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

HERE = Path(__file__).resolve().parent
STAGE3_ROOT = HERE.parent
PROJECT_ROOT = STAGE3_ROOT.parent
DATA_PATH = PROJECT_ROOT / "Dataset" / "prepped_hospital_data.csv"
V1_PARAMS = STAGE3_ROOT / "v1_xgboost" / "best_params.json"
V2_PARAMS = STAGE3_ROOT / "v2_xgboost" / "best_params.json"
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

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


def evaluate(name, probs, y_test, threshold):
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_test, preds)
    return {
        "model": name,
        "test_roc_auc": float(roc_auc_score(y_test, probs)),
        "test_pr_auc": float(average_precision_score(y_test, probs)),
        "test_brier": float(brier_score_loss(y_test, probs)),
        "test_f2": float(fbeta_score(y_test, preds, beta=2)),
        "threshold": float(threshold),
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        "probs": probs, "preds": preds,
    }


def train_logreg(Xtr, ytr, Xv, yv, Xte):
    pipe = Pipeline(steps=[
        ("preprocess", build_preprocessor()),
        ("logistic", LogisticRegression(max_iter=2000)),
    ])
    grid = {
        "logistic__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "logistic__l1_ratio": [0, 0.25, 0.5, 0.75, 1],
        "logistic__penalty": ["elasticnet"],
        "logistic__solver": ["saga"],
        "logistic__class_weight": [None, "balanced"],
    }
    g = GridSearchCV(pipe, grid, scoring="average_precision", cv=5, n_jobs=-1)
    g.fit(Xtr, ytr)
    val_probs = g.best_estimator_.predict_proba(Xv)[:, 1]
    thr = best_threshold(yv, val_probs)
    test_probs = g.best_estimator_.predict_proba(Xte)[:, 1]
    print(f"  LR  best CV PR-AUC: {g.best_score_:.4f}  best params: {g.best_params_}")
    return test_probs, thr


def _xgb_pipeline(best_params, scale_pos_weight):
    return Pipeline(steps=[
        ("preprocess", build_preprocessor()),
        ("xgb", XGBClassifier(
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", random_state=RANDOM_STATE,
            n_jobs=-1, scale_pos_weight=scale_pos_weight, **best_params,
        )),
    ])


def train_xgb(Xtr, ytr, Xv, yv, Xte, best_params, scale_pos_weight, calibrate=False, label="XGB"):
    pipe = _xgb_pipeline(best_params, scale_pos_weight)
    if calibrate:
        cal = CalibratedClassifierCV(estimator=pipe, method="isotonic", cv=5)
        cal.fit(Xtr, ytr)
        val_probs = cal.predict_proba(Xv)[:, 1]
        test_probs = cal.predict_proba(Xte)[:, 1]
    else:
        pipe.fit(Xtr, ytr)
        val_probs = pipe.predict_proba(Xv)[:, 1]
        test_probs = pipe.predict_proba(Xte)[:, 1]
    thr = best_threshold(yv, val_probs)
    print(f"  {label}  val PR-AUC = {average_precision_score(yv, val_probs):.4f}")
    return test_probs, thr


def _coerce_xgb_params(raw):
    int_keys = {"n_estimators", "max_depth", "min_child_weight", "max_delta_step"}
    out = {}
    for k, v in raw.items():
        if k in int_keys:
            out[k] = int(v)
        else:
            out[k] = float(v)
    return out


def main() -> None:
    Xtr, Xv, Xte, ytr, yv, yte = load_splits()
    print(f"Splits  train={len(Xtr)}  val={len(Xv)}  test={len(Xte)}")

    print("\n[1/3] Tuning Logistic Regression (GridSearchCV) ...")
    lr_test_probs, lr_thr = train_logreg(Xtr, ytr, Xv, yv, Xte)

    v1_meta = json.loads(V1_PARAMS.read_text())
    v1_params = _coerce_xgb_params(v1_meta["best_params"])
    scale_pos_weight = float(v1_meta["scale_pos_weight"])
    print("\n[2/3] Refitting XGBoost v1 with saved best params ...")
    v1_test_probs, v1_thr = train_xgb(
        Xtr, ytr, Xv, yv, Xte, v1_params, scale_pos_weight,
        calibrate=False, label="XGB v1",
    )

    v2_meta = json.loads(V2_PARAMS.read_text())
    v2_params = _coerce_xgb_params(v2_meta["best_params"])
    print("\n[3/3] Refitting XGBoost v2 with saved best params + isotonic ...")
    v2_test_probs, v2_thr = train_xgb(
        Xtr, ytr, Xv, yv, Xte, v2_params, scale_pos_weight,
        calibrate=True, label="XGB v2",
    )

    results = [
        evaluate("Logistic Regression (tuned)", lr_test_probs, yte, lr_thr),
        evaluate("XGBoost v1 (tuned)", v1_test_probs, yte, v1_thr),
        evaluate("XGBoost v2 (tuned + isotonic)", v2_test_probs, yte, v2_thr),
    ]

    rows = [{k: v for k, v in r.items() if k not in ("probs", "preds")} for r in results]
    summary = pd.DataFrame(rows).sort_values("test_roc_auc", ascending=False)
    summary.to_csv(HERE / "metrics.csv", index=False)

    print("\n=== FINAL TEST SET COMPARISON (sorted by ROC-AUC) ===")
    print(summary.to_string(index=False))
    winner = summary.iloc[0]
    print(f"\nWinner by test ROC-AUC: {winner['model']}")
    print(f"  ROC-AUC={winner['test_roc_auc']:.4f}  PR-AUC={winner['test_pr_auc']:.4f}  "
          f"Brier={winner['test_brier']:.4f}  F2={winner['test_f2']:.4f}")

    plt.figure(figsize=(7.5, 6.0))
    for r in results:
        fpr, tpr, _ = roc_curve(yte, r["probs"])
        plt.plot(fpr, tpr, label=f"{r['model']} (AUC={r['test_roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="grey", label="Chance")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("Stage 3 final - Test ROC (LR vs XGBoost v1 vs XGBoost v2)")
    plt.legend(loc="lower right", fontsize=9); plt.tight_layout()
    plt.savefig(FIG_DIR / "roc.png", dpi=150); plt.close()

    base_rate = float(np.mean(yte))
    plt.figure(figsize=(7.5, 6.0))
    for r in results:
        p, rec, _ = precision_recall_curve(yte, r["probs"])
        plt.plot(rec, p, label=f"{r['model']} (AP={r['test_pr_auc']:.3f})")
    plt.hlines(base_rate, 0, 1, colors="grey", linestyles="--",
               label=f"Base rate = {base_rate:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Stage 3 final - Test Precision-Recall")
    plt.legend(loc="lower left", fontsize=9); plt.tight_layout()
    plt.savefig(FIG_DIR / "pr.png", dpi=150); plt.close()

    fig, axes = plt.subplots(1, len(results), figsize=(4.6 * len(results), 4.5))
    for ax, r in zip(axes, results):
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{r['model']}\nthr={r['threshold']:.3f}", fontsize=9)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["True 0", "True 1"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="black" if cm[i, j] < cm.max() / 2 else "white")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Stage 3 final - Test confusion matrices (calibrated thresholds)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_grid.png", dpi=150); plt.close(fig)

    plt.figure(figsize=(7.0, 6.0))
    plt.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    for r in results:
        prob_true, prob_pred = calibration_curve(yte, r["probs"], n_bins=10, strategy="quantile")
        plt.plot(prob_pred, prob_true, marker="o",
                 label=f"{r['model']} (Brier={r['test_brier']:.3f})")
    plt.xlabel("Predicted probability"); plt.ylabel("Observed positive rate")
    plt.title("Stage 3 final - Test calibration")
    plt.legend(loc="lower right", fontsize=9); plt.tight_layout()
    plt.savefig(FIG_DIR / "calibration.png", dpi=150); plt.close()

    v1_row = next(r for r in results if r["model"].startswith("XGBoost v1"))
    v2_row = next(r for r in results if r["model"].startswith("XGBoost v2"))

    with open(HERE / "results.txt", "w") as f:
        f.write("=== Stage 3 final 3-way comparison: LR vs XGBoost v1 vs XGBoost v2 ===\n\n")
        f.write(summary.to_string(index=False))
        f.write("\n\n")
        f.write(f"Winner by test ROC-AUC: {winner['model']}\n\n")

        f.write("--- v1 vs v2 head-to-head (held-out test set) ---\n")
        f.write(f"{'Metric':<14} {'v1':>10} {'v2':>10} {'Delta':>10}\n")
        f.write(f"{'ROC-AUC':<14} {v1_row['test_roc_auc']:>10.4f} "
                f"{v2_row['test_roc_auc']:>10.4f} "
                f"{v2_row['test_roc_auc'] - v1_row['test_roc_auc']:>+10.4f}\n")
        f.write(f"{'PR-AUC':<14} {v1_row['test_pr_auc']:>10.4f} "
                f"{v2_row['test_pr_auc']:>10.4f} "
                f"{v2_row['test_pr_auc'] - v1_row['test_pr_auc']:>+10.4f}\n")
        f.write(f"{'Brier':<14} {v1_row['test_brier']:>10.4f} "
                f"{v2_row['test_brier']:>10.4f} "
                f"{v2_row['test_brier'] - v1_row['test_brier']:>+10.4f}  (lower=better)\n")
        f.write(f"{'F2':<14} {v1_row['test_f2']:>10.4f} "
                f"{v2_row['test_f2']:>10.4f} "
                f"{v2_row['test_f2'] - v1_row['test_f2']:>+10.4f}\n")
        f.write(f"{'Threshold':<14} {v1_row['threshold']:>10.4f} "
                f"{v2_row['threshold']:>10.4f}\n\n")

        for r in results:
            f.write(f"--- {r['model']} ---\n")
            f.write(f"threshold={r['threshold']:.4f}\n")
            f.write(classification_report(yte, r["preds"], digits=3))
            f.write(f"Confusion matrix: TN={r['tn']} FP={r['fp']} "
                    f"FN={r['fn']} TP={r['tp']}\n\n")


if __name__ == "__main__":
    main()
