# Stage 2 — Baseline Model Comparison Report

**Author:** Himanshu Singh Rao
**Date:** 25 April 2026
**Task (from proposal):** *Prototype 1–2 baseline models (e.g., logistic regression and one tree-based model); report validation ROC–AUC and precision–recall so the group can compare approaches before locking a final model.*

This report fulfills Himanshu's Stage 2 deliverable. The full reproducible
analysis lives in
[`stage2_baseline_comparison.ipynb`](stage2_baseline_comparison.ipynb)
(run inside the project-root `venv`).

---

## 1. How this is different from Ramani's and Alejandro's Stage 2 tasks

| Teammate | Stage 2 task (proposal) | What's on `main` for it | What Himanshu's Stage 2 adds |
|---|---|---|---|
| Alejandro | *Engineer clinically meaningful features (prior admission, length of stay, etc.)* | Feature-engineering work, independent of modeling. | Himanshu's notebook **does not** add new features — it stays inside the modeling lane. |
| Ramani | *Prototype 1–2 baseline models: logistic, random forest, gradient boosting, etc..* | `Model_1.py` (LogReg) + `Model_2.py` (Random Forest). Each script trains **one** model in isolation and prints its own metrics. | This work **unifies** the prototypes: identical splits, identical preprocessing, side-by-side ROC/PR/confusion-matrix figures, a metrics table, and a Stage 3 recommendation. |
| Himanshu | *Prototype baselines + **report ROC–AUC and precision–recall so the group can compare approaches before locking a final model.*** | `Himanshu_Stage2/` (this folder). | The **comparison/“scoreboard”** layer of Stage 2, plus a stronger candidate model (**XGBoost**) recommended for Stage 3 tuning. |

In short: Ramani built two isolated baseline scripts. Himanshu's job is the
comparison report sitting on top, plus pushing toward a stronger Stage 3
candidate.

## 2. Setup

- **Data:** `Dataset/prepped_hospital_data.csv` (8,000 patient encounters, target `label` = readmitted within 30 days)
- **Split:** 70 / 15 / 15 train / validation / test, stratified on `label`, `random_state = 42`
- **Sizes:** train = 5,600 · val = 1,200 · test = 1,200
- **Positive rate:** train 0.773 · val 0.773 · test 0.772
  - Note: the data is imbalanced *toward* the positive class. PR-AUC is reported alongside ROC-AUC, and absolute precision/recall numbers are read against the 0.773 base rate, not 0.5.
- **Preprocessing:** `OneHotEncoder(handle_unknown="ignore")` for categorical columns; `StandardScaler` for numeric columns. `gender` is already 0/1 in the prepped CSV and passes through.
- **Environment:** project-root `venv/` from `requirements.txt`.

## 3. Models prototyped

The proposal asked for "logistic regression and one tree-based model." Both
are present, plus a stretch baseline (Gradient Boosting) and the recommended
Stage 3 candidate (XGBoost).

| # | Model | Library / class | Notable settings |
|---|-------|-----------------|------------------|
| 1 | Logistic Regression | `sklearn.linear_model.LogisticRegression` | `max_iter=2000`, `class_weight="balanced"`, `random_state=42` |
| 2 | Random Forest (tree-based) | `sklearn.ensemble.RandomForestClassifier` | `n_estimators=300`, `class_weight="balanced"`, `random_state=42`, `n_jobs=-1` |
| 3 | Gradient Boosting (stretch) | `sklearn.ensemble.GradientBoostingClassifier` | `random_state=42`, defaults |
| 4 | **XGBoost (recommended)** | `xgboost.XGBClassifier` | `n_estimators=500`, `learning_rate=0.05`, `max_depth=5`, `min_child_weight=2`, `subsample=0.9`, `colsample_bytree=0.9`, `reg_lambda=1.0`, `tree_method="hist"`, `scale_pos_weight = neg/pos (train)`, `eval_metric="logloss"` |

All four are wrapped in the same `Pipeline(preprocess → clf)` so the
comparison is apples-to-apples.

## 4. Validation results

Sorted by ROC-AUC (highest first).

| Model | ROC-AUC | PR-AUC | Precision (1) | Recall (1) | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression (class-weighted) | **0.8580** | **0.9485** | 0.938 | 0.750 | 696 |  46 | 226 | 232 |
| Gradient Boosting                    | 0.8461     | 0.9448     | 0.846 | 0.928 | 861 | 157 | 115 |  67 |
| **XGBoost (recommended for Stage 3)**| 0.8367     | 0.9437     | 0.907 | 0.779 | 723 |  74 | 198 | 205 |
| Random Forest                        | 0.8295     | 0.9437     | 0.835 | 0.930 | 863 | 171 | 101 |  65 |

Held-out test results (reported once for sanity, not iterated on):

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Logistic Regression (class-weighted) | 0.8312 | 0.9372 |
| Gradient Boosting                    | 0.8248 | 0.9355 |
| XGBoost (recommended)                | 0.8116 | 0.9330 |
| Random Forest                        | 0.8080 | 0.9303 |

Test trends mirror validation, so the validation ranking is trustworthy.

### Curves & confusion matrices (validation)

- `figures/roc_curves_validation.png` — overlaid ROC curves
- `figures/pr_curves_validation.png` — overlaid PR curves with the 0.773 base-rate reference
- `figures/confusion_matrices_validation.png` — side-by-side confusion matrices at the default 0.5 threshold
- `figures/xgboost_top_features.png` — top-15 XGBoost gain-based importances

## 5. Reading the numbers

- All four models cluster within ≈ 0.03 ROC-AUC of each other and ≈ 0.005 PR-AUC. With this prepared feature set — a small set of mostly monotonic clinical/utilization predictors — a well-regularized linear model is genuinely hard to beat, which matches what the Stage 1 literature review predicted for LACE/HOSPITAL-style inputs.
- **Logistic Regression with `class_weight="balanced"`** leads on both validation ROC-AUC and PR-AUC. The balanced weighting is what shifts its operating point so that its negative-class recall is much higher than the unweighted baselines on `main`.
- **XGBoost** lands within 0.02 ROC-AUC of the leader using only conservative, hand-picked settings (no tuning, no early stopping). It is recommended as the second Stage 3 candidate because it has by far the richest tuning surface — `learning_rate`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `reg_lambda`, `gamma`, plus early stopping — and stands the best chance of pulling ahead once those are tuned in Stage 3.
- At the default 0.5 threshold every model shows the expected behavior on a 77% positive-class problem. Threshold tuning is a Stage 3 task, not a model-choice issue.

## 6. Recommendation for Stage 3

1. **Carry forward two models:** class-weighted **Logistic Regression** (current leader; interpretable; easy to defend in Stage 4) and **XGBoost** (best tuning candidate).
2. **Drop default-settings Random Forest and Gradient Boosting** unless they are revisited with tuned settings.
3. **Stage 3 tuning** (Himanshu's Stage 3 task):
   - XGBoost: stratified CV grid / random search over `n_estimators`, `learning_rate`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `reg_lambda`, `gamma`; early stopping on the validation split.
   - Logistic Regression: tune `C` and `penalty` (`l1` / `l2` / `elasticnet`) with stratified CV.
4. **Threshold calibration** from a precision-recall curve before any test-set scoring.
5. **Selection metric:** validation ROC-AUC primary, PR-AUC tie-breaker. Test set stays untouched until the final model is locked.

## 7. Files produced

```
Himanshu_Stage2/
├── stage2_baseline_comparison.ipynb   # primary deliverable (executed)
├── baseline_comparison.py             # equivalent pure-Python script
├── run_output.txt                     # captured stdout from the .py script
├── metrics_validation.csv             # per-model validation metrics
├── metrics_test.csv                   # per-model held-out test metrics
├── stage2_baseline_report.md          # this report
└── figures/
    ├── roc_curves_validation.png
    ├── pr_curves_validation.png
    ├── confusion_matrices_validation.png
    └── xgboost_top_features.png
```

### How to reproduce

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute \
    Himanshu_Stage2/stage2_baseline_comparison.ipynb --inplace
# ...or open the notebook in JupyterLab:
# jupyter lab Himanshu_Stage2/stage2_baseline_comparison.ipynb
```
