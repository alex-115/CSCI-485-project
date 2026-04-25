"""
Stage 2 — Himanshu Singh Rao
Baseline model prototyping for 30-day readmission prediction.

Per the proposal, this script prototypes 1-2 baseline models (logistic
regression and a tree-based model) and reports validation ROC-AUC and
precision-recall (PR-AUC), so the team can compare approaches before
locking a final model in Stage 3.

Outputs (under Himanshu_Stage2/):
    figures/roc_curves_validation.png
    figures/pr_curves_validation.png
    figures/confusion_matrices_validation.png
    metrics_validation.csv
    metrics_test.csv
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DATA_PATH = PROJECT_ROOT / "Dataset" / "prepped_hospital_data.csv"
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

NUMERIC_COLS = [
    "age",
    "comorbidities_count",
    "length_of_stay",
    "medications_count",
    "followup_visits_last_year",
    "prev_readmissions",
]

CATEGORICAL_COLS = [
    "season",
    "region",
    "primary_diagnosis",
    "treatment_type",
    "insurance_type",
    "discharge_disposition",
]


# ---------------------------------------------------------------------------
# Data loading + splitting (70 / 15 / 15, stratified on the label)
# ---------------------------------------------------------------------------

def load_splits(path: Path):
    df = pd.read_csv(path)
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
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
            ("num", StandardScaler(), NUMERIC_COLS),
        ],
        remainder="passthrough",
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def build_models() -> dict[str, Pipeline]:
    """The two baselines named in Himanshu's task plus a stretch baseline."""
    return {
        "Logistic Regression": Pipeline(steps=[
            ("preprocess", build_preprocessor()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline(steps=[
            ("preprocess", build_preprocessor()),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
        "Gradient Boosting": Pipeline(steps=[
            ("preprocess", build_preprocessor()),
            ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]),
    }


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate(name: str, model: Pipeline, X, y) -> dict:
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)
    roc = roc_auc_score(y, probs)
    pr = average_precision_score(y, probs)
    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
    return {
        "model": name,
        "roc_auc": roc,
        "pr_auc": pr,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision_pos": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall_pos": tp / (tp + fn) if (tp + fn) else 0.0,
        "probs": probs,
        "preds": preds,
    }


def plot_roc(results: list[dict], y_true, save_to: Path) -> None:
    plt.figure(figsize=(7, 6))
    for r in results:
        fpr, tpr, _ = roc_curve(y_true, r["probs"])
        plt.plot(fpr, tpr, label=f"{r['model']} (AUC = {r['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Validation ROC Curves — 30-Day Readmission")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_to, dpi=150)
    plt.close()


def plot_pr(results: list[dict], y_true, save_to: Path) -> None:
    base_rate = float(np.mean(y_true))
    plt.figure(figsize=(7, 6))
    for r in results:
        precision, recall, _ = precision_recall_curve(y_true, r["probs"])
        plt.plot(recall, precision, label=f"{r['model']} (AP = {r['pr_auc']:.3f})")
    plt.hlines(base_rate, 0, 1, colors="grey", linestyles="--",
               label=f"Base rate = {base_rate:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Validation Precision-Recall Curves — 30-Day Readmission")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_to, dpi=150)
    plt.close()


def plot_confusion_grid(results: list[dict], y_true, save_to: Path) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.5))
    if len(results) == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{r['model']}\nROC-AUC={r['roc_auc']:.3f}  PR-AUC={r['pr_auc']:.3f}")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["True 0", "True 1"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="black" if cm[i, j] < cm.max() / 2 else "white")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Validation Confusion Matrices (default 0.5 threshold)")
    fig.tight_layout()
    fig.savefig(save_to, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Loading data from {DATA_PATH.relative_to(PROJECT_ROOT)}")
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits(DATA_PATH)

    print(
        f"Split sizes  train={len(X_train)}  val={len(X_val)}  test={len(X_test)}\n"
        f"Positive rate train={y_train.mean():.3f}  val={y_val.mean():.3f}  test={y_test.mean():.3f}"
    )

    models = build_models()

    val_results, test_results = [], []
    for name, pipe in models.items():
        print(f"\n=== {name} ===")
        pipe.fit(X_train, y_train)

        val_eval = evaluate(name, pipe, X_val, y_val)
        test_eval = evaluate(name, pipe, X_test, y_test)
        val_results.append(val_eval)
        test_results.append(test_eval)

        print(f"Validation  ROC-AUC = {val_eval['roc_auc']:.4f}   PR-AUC = {val_eval['pr_auc']:.4f}")
        print("Validation classification report:")
        print(classification_report(y_val, val_eval["preds"], digits=3))

    val_df = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ("probs", "preds")} for r in val_results
    ])
    test_df = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ("probs", "preds")} for r in test_results
    ])
    val_df.to_csv(HERE / "metrics_validation.csv", index=False)
    test_df.to_csv(HERE / "metrics_test.csv", index=False)

    plot_roc(val_results, y_val, FIG_DIR / "roc_curves_validation.png")
    plot_pr(val_results, y_val, FIG_DIR / "pr_curves_validation.png")
    plot_confusion_grid(val_results, y_val, FIG_DIR / "confusion_matrices_validation.png")

    print("\n=== Validation summary (sorted by ROC-AUC) ===")
    print(val_df.sort_values("roc_auc", ascending=False).to_string(index=False))
    print("\n=== Test summary (sorted by ROC-AUC) ===")
    print(test_df.sort_values("roc_auc", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
