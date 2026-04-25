# Stage 2 — Himanshu Singh Rao Task

**Project:** 30-Day Hospital Readmission Risk Prediction (CSCI-485)
**Branch:** `stage2`
**Author:** Himanshu Singh Rao
**Date:** 25 April 2026

This document records Himanshu's Stage 2 deliverable, **explains how it
differs from Ramani's and Alejandro's Stage 2 tasks**, summarizes the
**stronger-model recommendation** carried into Stage 3, and cites the
Stage 1 work via `git log`.

---

## 1. Task statement (from `proposal.txt`)

> **Stage 2 task(s):** Prototype 1–2 baseline models (e.g., logistic regression
> and one tree-based model); **report validation ROC–AUC and precision–recall
> so the group can compare approaches before locking a final model.**

## 2. How Himanshu's Stage 2 differs from his teammates'

The proposal splits Stage 2 across three teammates with overlapping but
distinct deliverables. The differentiator for Himanshu is the *reporting /
comparison* clause at the end of his task.

| Teammate | Stage 2 task (proposal) | What lives on `main` for it | What this stage2 branch adds beyond that |
|---|---|---|---|
| **Alejandro** | *Engineer clinically meaningful features (prior admission, length of stay, etc.)* | Feature engineering work, independent of any modeling. | Himanshu's Stage 2 work **does not** touch feature engineering — it stays inside the modeling lane and consumes Alejandro's prepped CSV as-is. |
| **Ramani** | *Prototype 1–2 baseline models: logistic, random forest, gradient boosting, etc..* | `Model_1.py` (LogReg) and `Model_2.py` (Random Forest) — each script trains **one** model on its own copy of the data and prints its own metrics. There is no shared comparison. | Himanshu's notebook **unifies** the prototypes: identical 70/15/15 stratified split, identical preprocessing pipeline, side-by-side ROC / PR / confusion-matrix figures, machine-readable metrics CSVs. |
| **Himanshu (this branch)** | *Prototype 1–2 baselines + **report ROC–AUC and PR–AUC so the group can compare approaches before locking a final model.*** | `Himanshu_Stage2/` (this branch only). | The **comparison/“scoreboard”** layer of Stage 2, plus a stronger candidate model (**XGBoost**, class-weighted) recommended for Stage 3 tuning. |

In one sentence: **Ramani prototyped two baseline models; Himanshu
prototyped four (including a stronger candidate) and produced the
side-by-side comparison report the proposal asks for.**

## 3. Deliverables on the `stage2` branch

All Stage 2 artifacts live under `Himanshu_Stage2/`. The notebook is the
primary deliverable; the `.py` script is an equivalent pure-Python version
for non-notebook users.

| File | Purpose |
|------|---------|
| **`Himanshu_Stage2/stage2_baseline_comparison.ipynb`** | **Primary deliverable.** Reproducible Jupyter notebook: data load → 70/15/15 split → shared preprocessing → train all four models → tables → ROC / PR / confusion-matrix plots → XGBoost feature importances → written Stage 3 recommendation. Executed end-to-end inside the project-root `venv`. |
| `Himanshu_Stage2/baseline_comparison.py` | Equivalent pure-Python script (subset of the notebook, no XGBoost) — handy if anyone wants to run it without Jupyter. |
| `Himanshu_Stage2/metrics_validation.csv` | Per-model validation ROC-AUC, PR-AUC, precision, recall, and confusion-matrix counts. |
| `Himanshu_Stage2/metrics_test.csv` | Same metrics on the held-out test split (kept untouched; reported once for sanity). |
| `Himanshu_Stage2/run_output.txt` | Captured stdout from the `.py` run. |
| `Himanshu_Stage2/figures/roc_curves_validation.png` | Overlaid ROC curves for all four models. |
| `Himanshu_Stage2/figures/pr_curves_validation.png` | Overlaid precision-recall curves with the 0.773 base-rate reference line. |
| `Himanshu_Stage2/figures/confusion_matrices_validation.png` | Side-by-side validation confusion matrices at the default 0.5 threshold. |
| `Himanshu_Stage2/figures/xgboost_top_features.png` | Top-15 XGBoost gain-based importances (sanity check). |
| `Himanshu_Stage2/stage2_baseline_report.md` | Written analysis + Stage 3 recommendation. |
| `requirements.txt` (project root) | Reproducible dependency list. |
| `venv/` (project root, gitignored) | Local virtual environment used to run everything. |

## 4. Models prototyped

| # | Model | Why it's here |
|---|-------|---------------|
| 1 | **Logistic Regression** (class-weighted) | The "logistic regression" baseline named in Himanshu's task. |
| 2 | **Random Forest** (tree-based) | The "one tree-based model" baseline named in Himanshu's task. |
| 3 | **Gradient Boosting** | Stretch baseline so the team has more than the minimum two for comparison. |
| 4 | **XGBoost** (`scale_pos_weight = neg/pos`) | **Better Stage 3 candidate** — richer tuning surface, regularization, histogram-based splits, and explicit handling of the imbalanced-aware evaluation the proposal calls out. |

## 5. Headline validation results (25 Apr 2026 run, executed in `venv/`)

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Logistic Regression (class-weighted) | **0.8580** | **0.9485** |
| Gradient Boosting | 0.8461 | 0.9448 |
| XGBoost (recommended for Stage 3) | 0.8367 | 0.9437 |
| Random Forest | 0.8295 | 0.9437 |

All four models cluster within ≈ 0.03 ROC-AUC. Class-weighted **Logistic
Regression** leads on this prepped feature set — consistent with what the
Stage 1 literature review predicted for LACE/HOSPITAL-style features —
while **XGBoost** is recommended as the second model to carry into Stage 3
because of its much richer tuning surface and likely upside under proper
hyperparameter search + early stopping.

Full numbers, classification reports, and curves are in
[`Himanshu_Stage2/stage2_baseline_report.md`](Himanshu_Stage2/stage2_baseline_report.md)
and the executed notebook.

## 6. The "better model" recommendation

The team should adopt **two finalists for Stage 3**, not one:

1. **Class-weighted Logistic Regression** — current leader on both metrics, easy to defend in Stage 4 interpretability work.
2. **XGBoost** — the stronger *tuning* candidate. Even with conservative hand-picked settings (`learning_rate=0.05`, `max_depth=5`, `n_estimators=500`, `scale_pos_weight` from the training distribution) it sits within 0.02 ROC-AUC of the leader; tuning + early stopping is expected to close or invert that gap.

Drop default-settings Random Forest and Gradient Boosting unless they are
revisited with tuned settings. Threshold calibration from a precision-recall
curve is also a Stage 3 task — none of these numbers should be read against
a 0.5 cut.

## 7. Reproducibility

```bash
git checkout stage2
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

jupyter nbconvert --to notebook --execute \
    Himanshu_Stage2/stage2_baseline_comparison.ipynb --inplace
# or:
# jupyter lab Himanshu_Stage2/stage2_baseline_comparison.ipynb
```

Outputs are deterministic given `random_state=42`, the committed prepped
CSV, and the pinned ranges in `requirements.txt`.

---

## 8. Stage 1 task — completion citations

> **Stage 1 task(s):** Research hospital readmission and common clinical,
> demographic, and utilization predictors from the literature; share a short
> summary with the team listing variables and definitions worth prioritizing
> in our data.

Completed and merged into `main` before Stage 2 began. Citations
(verifiable with `git show <sha>`):

| SHA | Author | Date | Subject | Notes |
|-----|--------|------|---------|-------|
| `1d63954` | Himanshu Singh Rao | 2026-04-07 | *Himanshu's Stage 1 Task* | Added `stage1_variable_summary.md` — the team-facing Stage 1 deliverable listing demographic, clinical, and utilization variables with definitions, plus the LACE / HOSPITAL reference scoring tools. |
| `b4e68e6` | Alejandro (alex-115) | 2026-04-10 | *Merge pull request #1 from alex-115/stage1* | Merge commit that brought commit `1d63954` (Himanshu's Stage 1 work) onto `main` via the `stage1` branch. |

Verify locally with:

```bash
git log --oneline 1d63954 -1
git show --stat 1d63954
git log --oneline b4e68e6 -1
```

The merged file (`stage1_variable_summary.md`) covers the demographic,
clinical, and utilization variables called for in the proposal and provides
the team a literature-grounded shortlist for Stages 2–4.

### Related Stage 1 work by other teammates (for context)

| SHA | Author | Date | Subject | Owner / scope |
|-----|--------|------|---------|---------------|
| `b2bd998` | Alejandro (alex-115) | 2026-04-02 | *Finished Data preperation, look at code for more info* | Alejandro — Stage 1 data cleaning |
| `17634eb` | Sri Ramani Thungapati | 2026-04-04 | *EDA code added* | Ramani — Stage 1 exploratory data analysis |

---

## 9. How Stage 1 fed into Stage 2

The prepped CSV used by Himanshu's Stage 2 notebook
(`Dataset/prepped_hospital_data.csv`, produced by `Data_prep.py`) drops
identifiers and the leakage-prone `readmission_risk_score`, encodes `gender`
to 0/1, and keeps the variables that match the Stage 1 shortlist
(`age`, `comorbidities_count`, `length_of_stay`, `medications_count`,
`followup_visits_last_year`, `prev_readmissions`, plus the categorical
fields `season`, `region`, `primary_diagnosis`, `treatment_type`,
`insurance_type`, `discharge_disposition`). That alignment is what lets
Himanshu's notebook keep the modeling code short — heavier feature
engineering is Alejandro's Stage 2 task.

---

## 10. Branch / handoff notes

- This document and all Stage 2 artifacts are committed on the `stage2`
  branch only. `main` should not be modified by this work until the team
  agrees on the comparison.
- After review, merge `stage2` → `main` via PR (mirroring how Stage 1 was
  merged in `b4e68e6`).
- Stage 3 picks up from here: tune Logistic Regression and XGBoost,
  calibrate the decision threshold, and lock the final model.
