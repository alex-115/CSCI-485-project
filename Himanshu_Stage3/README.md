# Himanshu's Stage 3 - Folder Index

**Project:** 30-Day Hospital Readmission Risk Prediction (CSCI-485)
**Author:** Himanshu Singh Rao
**Stage 3 task (from `proposal.txt`):** *Lead training of the chosen final
model, optimization, and hyperparameter tuning using the validation set;
record the final settings and code path so results are reproducible.*

This folder is the entire Stage 3 deliverable. Everything in here is
**versioned by run** so anyone can map a metric back to the script and
hyperparameter set that produced it.

---

## Layout

```
Himanshu_Stage3/
├── README.md                     <- you are here
├── stage3_report.md              <- the Stage 3 written deliverable
├── xgboost_explained.md          <- detailed XGBoost vs LR walkthrough
│
├── v1_xgboost/                   <- run v1: tuned XGBoost, no calibration
│   ├── train.py                  <- the script that produces everything below
│   ├── best_params.json          <- frozen hyperparameters + metadata
│   ├── metrics.csv               <- one-row test-set metrics
│   ├── results.txt               <- captured stdout / classification report
│   └── figures/
│       ├── roc.png
│       ├── pr.png
│       ├── confusion.png
│       ├── calibration.png
│       └── top_features.png
│
├── v2_xgboost/                   <- run v2: wider search + isotonic calibration
│   ├── train.py
│   ├── best_params.json
│   ├── metrics.csv
│   ├── results.txt
│   └── figures/
│       ├── roc.png
│       ├── pr.png
│       └── calibration.png
│
└── final_comparison/             <- 3-way: LR + XGBoost v1 + XGBoost v2
    ├── compare.py
    ├── metrics.csv               <- one row per model, sorted by test ROC-AUC
    ├── results.txt               <- full text dump incl. v1 vs v2 head-to-head
    └── figures/
        ├── roc.png
        ├── pr.png
        ├── confusion_grid.png
        └── calibration.png
```

Every subfolder is **self-contained**: the training script, the
hyperparameters it produced, and the artifacts (`metrics.csv`,
`results.txt`, `figures/`) that script writes all live next to each
other. No cross-folder name collisions, no ambiguity about which run
produced which figure.

## Versioning convention

| Run | Folder | What's different vs the run before it |
|---|---|---|
| **v1** | `v1_xgboost/` | First tuned XGBoost: `RandomizedSearchCV(n_iter=40)`, 8 hyperparameters, no probability calibration. |
| **v2** | `v2_xgboost/` | Wider search: `n_iter=100`, +`reg_alpha`, +`colsample_bylevel`, +`max_delta_step`; lower `learning_rate` floor; up to 1500 estimators; **isotonic calibration** added on top of the tuned model. |
| **final** | `final_comparison/` | Re-fits v1 and v2 from their saved `best_params.json` and re-tunes Logistic Regression inline; produces the 3-way comparison + figures. |

Each `best_params.json` carries a `"version"` field so the file's
identity travels with it even if it's copied somewhere else.

The Logistic Regression baseline being compared against is the
optimized one already on `main`:
[`../Optimized_Logstic_model/Optimized_Logistic_model.py`](../Optimized_Logstic_model/Optimized_Logistic_model.py).
This Stage 3 work does not modify it.

## Reproducibility

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

`final_comparison/compare.py` does not re-tune XGBoost - it only reads
`v1_xgboost/best_params.json` and `v2_xgboost/best_params.json` and
refits. So if you want to update the comparison, **re-run the
corresponding training script first**, not just the comparison.

Outputs are deterministic given `random_state=42`, the committed
prepped CSV (`Dataset/prepped_hospital_data.csv`), and the pinned
packages in `requirements.txt`.

## Where to look for an answer

| If you're asking ... | Read this |
|---|---|
| "What did Stage 3 deliver?" | [`stage3_report.md`](stage3_report.md) |
| "What is XGBoost actually doing here, and why these hyperparameters?" | [`xgboost_explained.md`](xgboost_explained.md) |
| "What were the test-set numbers for each run?" | `*/results.txt` (per run), or `final_comparison/results.txt` (all together) |
| "Can the model be tuned further without overfitting?" | [`xgboost_explained.md` §8](xgboost_explained.md) and `v2_xgboost/results.txt` |
| "How do I reproduce a specific number?" | `*/best_params.json` carries the search method + hyperparameters; rerun the matching `train.py`. |
