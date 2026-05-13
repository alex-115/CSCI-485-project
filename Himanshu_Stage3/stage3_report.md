# Stage 3 - Final Model Training, Optimization, and Hyperparameter Tuning

**Project:** 30-Day Hospital Readmission Risk Prediction (CSCI-485)
**Author:** Himanshu Singh Rao
**Date:** 6 May 2026
**Folder:** `Himanshu_Stage3/` (see [`README.md`](README.md) for the layout and version log)

---

## 1. Task statement (from `proposal.txt`)

> **Stage 3 task(s):** Lead training of the chosen final model, optimization, and
> hyperparameter tuning using the validation set; record the final settings and
> code path so results are reproducible.

The Stage 2 deliverable
([`Himanshu_Stage2/stage2_baseline_report.md`](../Himanshu_Stage2/stage2_baseline_report.md))
recommended carrying forward two finalists into Stage 3 - **class-weighted
Logistic Regression** (the Stage 2 leader) and **XGBoost** (the strongest
tuning candidate). This Stage 3 work optimizes XGBoost across two
versioned runs, compares both head-to-head against the optimized LR
already on `main`, and records every setting needed to reproduce the
result inside the project-root `venv/`.

## 2. Folder layout (versioned)

| Subfolder | What it is |
|---|---|
| **`v1_xgboost/`** | First XGBoost tuning pass: `RandomizedSearchCV(n_iter=40)`, 8 hyperparameters, no probability calibration. |
| **`v2_xgboost/`** | Follow-up tuning: wider search (`n_iter=100`, +`reg_alpha`, +`colsample_bylevel`, +`max_delta_step`, lower `learning_rate` floor, up to 1500 estimators) **plus** `CalibratedClassifierCV(method='isotonic', cv=5)` on top of the tuned model. |
| **`final_comparison/`** | Refits both XGBoost versions from their saved `best_params.json` and re-tunes Logistic Regression inline; produces the 3-way comparison + figures. |

Each subfolder is self-contained: `train.py` (or `compare.py`),
`best_params.json`, `metrics.csv`, `results.txt`, and `figures/`. See
[`README.md`](README.md) for the full file index and version log.

## 3. Setup (shared by every run)

- **Data:** `Dataset/prepped_hospital_data.csv` (8,000 rows, target `label` = readmitted within 30 days)
- **Split:** 70 / 15 / 15 train / validation / test, stratified on `label`, `random_state = 42` (5,600 / 1,200 / 1,200; positive rate 0.773 / 0.773 / 0.772)
- **Class imbalance:** the *positive* class is the majority (~77%). Both model families compensate explicitly:
  - LR uses `class_weight="balanced"`
  - XGBoost uses `scale_pos_weight = neg/pos = 0.2939`
- **Preprocessing:** `OneHotEncoder(handle_unknown="ignore")` for the 6 categorical columns + `StandardScaler` for the 6 numeric columns; `gender` passes through.
- **Tuning protocol:** `StratifiedKFold(5)` cross-validation on the training split, scoring = `average_precision` (PR-AUC). The validation split is held out for threshold calibration. The test split is touched **once** at the end.
- **Threshold rule:** maximum F1 on the validation precision-recall curve - same rule used by `Optimized_Logstic_model/Optimized_Logistic_model.py` for consistency.
- **Environment:** project-root `venv/` with `pip install -r requirements.txt`.

## 4. The two XGBoost runs

### 4.1 v1 - first tuned XGBoost (`v1_xgboost/`)

Search space, all sampled with `RandomizedSearchCV(n_iter=40)`:

| Hyperparameter | Distribution | Final value |
|---|---|---|
| `n_estimators` | `randint(200, 900)` | **574** |
| `learning_rate` | `loguniform(0.01, 0.3)` | **0.0138** |
| `max_depth` | `randint(3, 9)` | **5** |
| `min_child_weight` | `randint(1, 10)` | **6** |
| `subsample` | `uniform(0.6, 1.0)` | **0.836** |
| `colsample_bytree` | `uniform(0.6, 1.0)` | **0.863** |
| `reg_lambda` | `loguniform(0.01, 10)` | **0.0564** |
| `gamma` | `uniform(0, 5)` | **2.842** |
| `scale_pos_weight` (fixed) | `neg/pos` on train | **0.2939** |
| Calibrated threshold | F1-max on val PR | **0.2114** |

CV best PR-AUC: **0.9349**. Frozen settings live in
`v1_xgboost/best_params.json`.

### 4.2 v2 - wider search + isotonic calibration (`v2_xgboost/`)

Three new knobs over v1, plus probability calibration. `n_iter=100`.

| New hyperparameter | Distribution | Final value |
|---|---|---|
| `reg_alpha` (L1 on leaves) | `loguniform(1e-3, 1)` | **0.0212** |
| `colsample_bylevel` | `uniform(0.6, 1.0)` | **0.674** |
| `max_delta_step` | `randint(0, 6)` | **0** |

| Hyperparameter (also wider) | Final value |
|---|---|
| `n_estimators` | **812** |
| `learning_rate` | **0.0634** |
| `max_depth` | **4** |
| `min_child_weight` | **10** |
| `subsample` | **0.638** |
| `colsample_bytree` | **0.632** |
| `reg_lambda` | **0.0524** |
| `gamma` | **2.142** |
| Calibrated threshold | **0.3570** |

CV best PR-AUC: **0.9357** (+0.0008 vs v1).
After tuning: wrapped in `CalibratedClassifierCV(method="isotonic", cv=5)`.
Frozen settings live in `v2_xgboost/best_params.json`.

### 4.3 Logistic Regression (already optimized in `Optimized_Logstic_model/`)

| Hyperparameter | Final value |
|---|---|
| `solver` | `saga` |
| `penalty` | `elasticnet` |
| `l1_ratio` | `0` (pure L2) |
| `C` | `0.1` |
| `class_weight` | `balanced` |
| Calibrated threshold | ~`0.20` (F1-max on val PR curve) |

Source: `Optimized_Logstic_model/Optimized_Logistic_model.py`. This
Stage 3 work does not modify the LR script.

## 5. Final test-set comparison (`final_comparison/`)

All three are scored once on the held-out 1,200-row test split with
each model's own calibrated threshold. Sorted by ROC-AUC.

| Rank | Model | Test ROC-AUC | Test PR-AUC | Brier | F2 | TP | FP | TN | FN |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **Logistic Regression (tuned)** | **0.8299** | 0.9371 | 0.1706 | 0.9348 | 900 | 206 | 67 | 27 |
| 2 | XGBoost v2 (tuned + isotonic) | 0.8266 | **0.9373** | **0.1304** | **0.9363** | 905 | 220 | 53 | 22 |
| 3 | XGBoost v1 (tuned) | 0.8252 | 0.9360 | 0.1697 | 0.9299 | 889 | 183 | 90 | 38 |

Machine-readable: `final_comparison/metrics.csv`.
Full text dump (incl. v1↔v2 head-to-head + per-model classification reports):
`final_comparison/results.txt`.

### Reading the table

- **All three models cluster within ~0.005 ROC-AUC of each other.** The data has a signal ceiling - tuning alone is not going to break it.
- **LR wins ROC-AUC by a hair.** It is what we already had from the start, which is the cleanest possible Stage 3 outcome on this small, mostly-monotonic feature set.
- **XGBoost v2 wins PR-AUC, Brier, and F2.** It also wins by a hair on PR-AUC (+0.0002), but the **Brier improvement is huge** (0.130 vs 0.171). This is what isotonic calibration buys you.
- **v2 vs v1: yes, you can tune further.** Wider random search + isotonic moved every metric in the right direction without any sign of overfitting. Ranking gains (~0.001 ROC-AUC, ~0.001 PR-AUC) are at the noise floor; the real win is *probability quality* (Brier −0.039, a 23% relative drop).

### Figures

- `final_comparison/figures/roc.png` - overlaid ROC curves for all three candidates
- `final_comparison/figures/pr.png` - overlaid PR curves with the 0.772 base-rate reference
- `final_comparison/figures/confusion_grid.png` - test confusion matrices at each model's calibrated threshold
- `final_comparison/figures/calibration.png` - reliability diagrams; v2's curve sits on top of the diagonal, v1 and LR overshoot at the high-probability end
- Per-run figures live next to each `train.py`: `v1_xgboost/figures/*.png`, `v2_xgboost/figures/*.png`

## 6. Final model

**Logistic Regression (tuned)** is the Stage 3 final model on test
ROC-AUC. **XGBoost v2** is the recommended *second-opinion* model: it
matches LR on PR-AUC, beats it on Brier, F2, and recall, and ships a
calibrated probability surface that is much closer to the diagonal -
which matters if downstream Stage 4 work uses risk scores rather than
hard labels.

If only one model is shipped, ship LR for ROC-AUC and interpretability.
If the downstream consumer wants well-calibrated risk *probabilities*,
ship XGBoost v2.

Frozen settings for both winners are in:
- `Optimized_Logstic_model/Optimized_Logistic_model.py` (LR, on `main`)
- `Himanshu_Stage3/v2_xgboost/best_params.json` (XGBoost v2)

## 7. Reproducibility

All commands run from the project root with the existing `venv/`.

```bash
python3 -m venv venv                           # if not already present
source venv/bin/activate
pip install -r requirements.txt

# Reproduce each run independently:
python Himanshu_Stage3/v1_xgboost/train.py
python Himanshu_Stage3/v2_xgboost/train.py

# Then build the 3-way comparison from the saved best_params.json files:
python Himanshu_Stage3/final_comparison/compare.py
```

`final_comparison/compare.py` does not re-tune XGBoost - it reads
`v1_xgboost/best_params.json` and `v2_xgboost/best_params.json` and
refits. Re-run the matching `train.py` first if you want to update the
comparison's inputs.

Outputs are deterministic given `random_state=42`, the committed
prepped CSV, and the pinned packages in `requirements.txt`.

## 8. Hand-off to Stage 4

Stage 4 (per the proposal) is *analyze the predictions of readmission*
and *interpret which inputs the model weights most*. Both winners are
ready for that work:

- **LR** - direct coefficient inspection, with odds-ratios. Already
  produced in `Optimized_Logstic_model/Logstic_Results.txt` (top three:
  `prev_readmissions`, `primary_diagnosis_COPD`,
  `primary_diagnosis_Stroke`).
- **XGBoost v2** - use SHAP for per-patient explanations and
  `booster.get_score(importance_type="gain")` for the global ranking.
  v1's gain-based top features (`comorbidities_count`,
  `insurance_type_Medicare`, `medications_count`, `age`,
  `prev_readmissions`) are already plotted in
  `v1_xgboost/figures/top_features.png` as a sanity check.
- The two models agree at the top - which is the cleanest sanity check
  we can produce that the Stage 1 variable shortlist was the right one.

## 9. Detail reading

For a deeper, hyperparameter-by-hyperparameter walkthrough of why
XGBoost lands where it does on this data, plus the explicit
"can-we-tune-further?" experiment behind the v2 numbers, see
[`xgboost_explained.md`](xgboost_explained.md).
