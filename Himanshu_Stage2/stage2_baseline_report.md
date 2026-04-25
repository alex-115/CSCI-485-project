# Stage 2 — Baseline Model Comparison Report

**Author:** Himanshu Singh Rao
**Date:** 25 April 2026
**Task (from proposal):** *Prototype 1–2 baseline models (e.g., logistic regression and one tree-based model); report validation ROC–AUC and precision–recall so the group can compare approaches before locking a final model.*

This report fulfills Himanshu's Stage 2 deliverable. The group should review the
metrics below before committing to a final model in Stage 3.

---

## 1. Setup

- **Data:** `Dataset/prepped_hospital_data.csv` (8,000 patient encounters, target `label` = readmitted within 30 days)
- **Split:** 70 / 15 / 15 train / validation / test, stratified on `label`, `random_state = 42`
- **Sizes:** train = 5,600 · validation = 1,200 · test = 1,200
- **Positive rate (label = 1):** train 0.773 · val 0.773 · test 0.772
  - Note: the dataset is imbalanced *toward* the positive class. PR-AUC is reported alongside ROC-AUC because the proposal explicitly calls for both, and the absolute precision/recall numbers should be read against the 0.773 base rate, not 0.5.
- **Preprocessing:** `OneHotEncoder(handle_unknown="ignore")` for categorical columns (`season`, `region`, `primary_diagnosis`, `treatment_type`, `insurance_type`, `discharge_disposition`) and `StandardScaler` for numeric columns (`age`, `comorbidities_count`, `length_of_stay`, `medications_count`, `followup_visits_last_year`, `prev_readmissions`). `gender` is already 0/1 in the prepped CSV and passes through.
- **Reproducibility:** All splits, fits, and evaluations live in `Himanshu_Stage2/baseline_comparison.py`. Re-running the script regenerates `metrics_validation.csv`, `metrics_test.csv`, and the figures under `Himanshu_Stage2/figures/`.

## 2. Models prototyped

The proposal asked for "logistic regression and one tree-based model." I ran both, plus a stretch baseline (gradient boosting) so the team has more than the minimum to choose from in Stage 3.

| # | Model | Library / class | Notable settings |
|---|-------|-----------------|------------------|
| 1 | Logistic Regression | `sklearn.linear_model.LogisticRegression` | `max_iter=1000`, `random_state=42` |
| 2 | Random Forest (tree-based) | `sklearn.ensemble.RandomForestClassifier` | `n_estimators=300`, `random_state=42`, `n_jobs=-1` |
| 3 | Gradient Boosting (stretch) | `sklearn.ensemble.GradientBoostingClassifier` | `random_state=42`, defaults |

All three are wrapped in the same `Pipeline(preprocess → clf)` so the comparison is apples-to-apples.

## 3. Validation results

Sorted by ROC-AUC (highest first).

| Model | ROC-AUC | PR-AUC (AP) | Precision (1) | Recall (1) | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | **0.8559** | **0.9477** | 0.834 | 0.952 | 883 | 176 | 96 | 45 |
| Gradient Boosting | 0.8461 | 0.9448 | 0.846 | 0.928 | 861 | 157 | 115 | 67 |
| Random Forest | 0.8281 | 0.9399 | 0.834 | 0.927 | 860 | 171 | 101 | 68 |

Held-out test results (for reference only — the team should not iterate on these):

| Model | ROC-AUC | PR-AUC (AP) |
|---|---:|---:|
| Logistic Regression | 0.8306 | 0.9369 |
| Gradient Boosting | 0.8248 | 0.9355 |
| Random Forest | 0.8069 | 0.9292 |

Test trends mirror validation, so the validation ranking is trustworthy.

### Curves & confusion matrices

- `figures/roc_curves_validation.png` — overlaid ROC curves
- `figures/pr_curves_validation.png` — overlaid PR curves with the 0.773 base-rate reference
- `figures/confusion_matrices_validation.png` — side-by-side confusion matrices at the default 0.5 threshold

## 4. Reading the numbers

- **ROC-AUC** is highest for **Logistic Regression (0.856)**, with Gradient Boosting close behind (0.846) and Random Forest a notch lower (0.828). All three are well above the 0.5 chance line.
- **PR-AUC** tells the same story (LR 0.948 → GB 0.945 → RF 0.940), and all three sit comfortably above the 0.773 base rate, meaning each model adds real signal over "always predict readmission."
- At the default 0.5 threshold every model is heavily biased toward the majority class: recall on the positive class is 0.93–0.95, but recall on the *negative* class (label = 0) is only 0.35–0.42. That is expected given the 77% positive rate; **threshold tuning and/or class-weighting will be a Stage 3 task**, not a model-choice issue.
- Logistic Regression generalizes well on this prepared feature set, which is consistent with what the Stage 1 literature review predicted for LACE/HOSPITAL-style features (see `stage1_variable_summary.md`): a small set of mostly monotonic clinical/utilization predictors tends to be well-served by a linear model.

## 5. Recommendation for Stage 3

1. **Carry forward Logistic Regression and Gradient Boosting** as the two finalists. They are within 0.01 ROC-AUC of each other and offer complementary strengths (interpretability vs. nonlinear interactions).
2. **Drop the default-settings Random Forest** unless we revisit it with tuned `max_depth` / `min_samples_leaf` in Stage 3.
3. **Stage 3 hyperparameter tuning** should focus on:
   - Logistic Regression: `C`, `penalty` (`l1` / `l2`), and class-weight balancing.
   - Gradient Boosting: `n_estimators`, `learning_rate`, `max_depth`, `subsample`.
4. **Threshold calibration**: since the application is "prioritize follow-up for high-risk patients," we should pick the operating threshold from a precision–recall trade-off curve rather than leaving it at 0.5.
5. **Evaluation metric** for Stage 3 selection: validation ROC-AUC as the primary metric, PR-AUC as the tie-breaker, with the test set untouched until the final model is locked.

## 6. Files produced

```
Himanshu_Stage2/
├── baseline_comparison.py        # reproducible script
├── run_output.txt                # captured stdout from the run on 25 Apr 2026
├── metrics_validation.csv        # per-model validation metrics
├── metrics_test.csv              # per-model held-out test metrics
├── stage2_baseline_report.md     # this report
└── figures/
    ├── roc_curves_validation.png
    ├── pr_curves_validation.png
    └── confusion_matrices_validation.png
```
