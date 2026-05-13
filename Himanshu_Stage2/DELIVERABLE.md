# Stage 2 — Main Deliverable

**Owner:** Himanshu Singh Rao
**Branch:** `stage2`
**Date:** 25 April 2026

---

## In one sentence

A reproducible Jupyter notebook that prototypes baseline models for 30-day
hospital readmission prediction, reports validation **ROC-AUC** and
**precision-recall** for each, and recommends two finalists to carry into
Stage 3 — fulfilling Himanshu's Stage 2 task in `proposal.txt`.

## The deliverable

> **`Himanshu_Stage2/stage2_baseline_comparison.ipynb`**

Open this notebook first. It is the single source of truth for the Stage 2
work. Everything else in this folder is either an input it consumes, an
output it produces, or written commentary about it.

## Headline result (validation, 25 Apr 2026 run)

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| **Logistic Regression (class-weighted)** | **0.8580** | **0.9485** |
| Gradient Boosting | 0.8461 | 0.9448 |
| **XGBoost (Stage 3 candidate)** | 0.8367 | 0.9437 |
| Random Forest | 0.8295 | 0.9437 |

**Recommendation for Stage 3:** carry **class-weighted Logistic Regression**
and **XGBoost** forward. Drop default-settings RF and GB. Calibrate the
decision threshold from a precision-recall curve before any test-set
scoring.

## Supporting artifacts (in this folder)

| File | What it is |
|------|-----------|
| `stage2_baseline_comparison.ipynb` | **The deliverable.** Executed end-to-end inside the project-root `venv/`. |
| `baseline_comparison.py` | Pure-Python equivalent of the notebook for non-Jupyter use. |
| `stage2_baseline_report.md` | Written analysis: methodology, results, Stage 3 recommendation. |
| `literature_review.md` | Survey of what other people have used for 30-day readmission prediction (UCI Diabetes 130-US benchmarks, synthetic-Kaggle benchmarks, recommended Stage 3 additions: CatBoost, LightGBM, ensemble). |
| `metrics_validation.csv`, `metrics_test.csv` | Per-model metrics (machine-readable). |
| `figures/roc_curves_validation.png` | Overlaid ROC curves. |
| `figures/pr_curves_validation.png` | Overlaid PR curves with the 0.773 base-rate reference. |
| `figures/confusion_matrices_validation.png` | Side-by-side confusion matrices at the default 0.5 threshold. |
| `figures/xgboost_top_features.png` | Top-15 XGBoost feature importances (sanity check). |
| `run_output.txt` | Captured stdout from the `.py` script. |

## How to reproduce

```bash
git checkout stage2
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

jupyter nbconvert --to notebook --execute \
    Himanshu_Stage2/stage2_baseline_comparison.ipynb --inplace
# or, interactively:
# jupyter lab Himanshu_Stage2/stage2_baseline_comparison.ipynb
```

Outputs are deterministic given `random_state=42`, the committed prepped
CSV (`Dataset/prepped_hospital_data.csv`), and the pinned ranges in
`requirements.txt`.

## Where to read more

- **Task statement + how this differs from Ramani's and Alejandro's Stage 2 tasks + Stage 1 git-log citations** → [`../stage2_himanshu_task.md`](../stage2_himanshu_task.md)
- **Full written analysis of the four-model comparison** → [`stage2_baseline_report.md`](stage2_baseline_report.md)
- **What other people have done (UCI + synthetic-Kaggle benchmarks, model recommendations)** → [`literature_review.md`](literature_review.md)
