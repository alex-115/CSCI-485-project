# Stage 2 — Himanshu Singh Rao Task

**Project:** 30-Day Hospital Readmission Risk Prediction (CSCI-485)
**Branch:** `stage2`
**Author:** Himanshu Singh Rao
**Date:** 25 April 2026

This document records Himanshu's Stage 2 deliverable for the project,
re-states what he completed in Stage 1 (with `git log` citations), and points
the rest of the team to the artifacts produced on the `stage2` branch.

---

## 1. Task statement (from `proposal.txt`)

> **Stage 2 task(s):** Prototype 1–2 baseline models (e.g., logistic regression
> and one tree-based model); report validation ROC–AUC and precision–recall so
> the group can compare approaches before locking a final model.

## 2. What was delivered on the `stage2` branch

All Stage 2 artifacts live under the `Himanshu_Stage2/` folder.

| File | Purpose |
|------|---------|
| `Himanshu_Stage2/baseline_comparison.py` | Reproducible script that loads `Dataset/prepped_hospital_data.csv`, makes a stratified 70/15/15 train/val/test split (seed = 42), wraps three models in identical preprocessing pipelines, and prints + saves validation and test metrics. |
| `Himanshu_Stage2/metrics_validation.csv` | Per-model validation ROC-AUC, PR-AUC, precision, recall, and confusion-matrix counts. |
| `Himanshu_Stage2/metrics_test.csv` | Same metrics on the held-out test split (kept untouched; reported once for sanity-checking generalization). |
| `Himanshu_Stage2/run_output.txt` | Captured stdout from the 25 Apr 2026 run, including full classification reports. |
| `Himanshu_Stage2/figures/roc_curves_validation.png` | Overlaid ROC curves for the three baselines. |
| `Himanshu_Stage2/figures/pr_curves_validation.png` | Overlaid precision-recall curves with the data's 0.773 base rate drawn in. |
| `Himanshu_Stage2/figures/confusion_matrices_validation.png` | Side-by-side validation confusion matrices at the default 0.5 threshold. |
| `Himanshu_Stage2/stage2_baseline_report.md` | Written analysis of the comparison and a recommendation for Stage 3. |

### Models prototyped

The proposal asked for *"logistic regression and one tree-based model"*. Both
are present, plus a stretch baseline so the team has more than the minimum
when picking a finalist:

1. **Logistic Regression** — `sklearn.linear_model.LogisticRegression(max_iter=1000)`
2. **Random Forest** (tree-based) — `sklearn.ensemble.RandomForestClassifier(n_estimators=300)`
3. **Gradient Boosting** (stretch) — `sklearn.ensemble.GradientBoostingClassifier()`

### Headline validation results (25 Apr 2026 run)

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Logistic Regression | **0.8559** | **0.9477** |
| Gradient Boosting | 0.8461 | 0.9448 |
| Random Forest | 0.8281 | 0.9399 |

**Recommendation to the team:** carry Logistic Regression and Gradient Boosting
forward into Stage 3 hyperparameter tuning. Full reasoning, threshold-tuning
notes, and Stage 3 follow-ups are in
[`Himanshu_Stage2/stage2_baseline_report.md`](Himanshu_Stage2/stage2_baseline_report.md).

### How to reproduce

```bash
git checkout stage2
python3 Himanshu_Stage2/baseline_comparison.py
```

Outputs are deterministic given `random_state=42` and the committed prepped
CSV.

---

## 3. Stage 1 task — completion citations

> **Stage 1 task(s):** Research hospital readmission and common clinical,
> demographic, and utilization predictors from the literature; share a short
> summary with the team listing variables and definitions worth prioritizing
> in our data.

This was completed and merged into `main` before Stage 2 began. Citations
(verifiable via `git log` / `git show <sha>`):

| SHA | Author | Date | Subject | Notes |
|-----|--------|------|---------|-------|
| [`1d63954`](#) | Himanshu Singh Rao | 2026-04-07 | *Himanshu's Stage 1 Task* | Added `stage1_variable_summary.md` — the team-facing Stage 1 deliverable listing demographic, clinical, and utilization variables with definitions, plus the LACE / HOSPITAL reference scoring tools. |
| [`b4e68e6`](#) | Alejandro (alex-115) | 2026-04-10 | *Merge pull request #1 from alex-115/stage1* | Merge commit that brought commit `1d63954` (Himanshu's Stage 1 work) onto `main` via the `stage1` branch. |

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

These commits land on `main` alongside Himanshu's Stage 1 deliverable; they
are owned by other teammates per the proposal's task decomposition but are
useful to cite together so the Stage 1 picture is complete:

| SHA | Author | Date | Subject | Owner / scope |
|-----|--------|------|---------|---------------|
| `b2bd998` | Alejandro (alex-115) | 2026-04-02 | *Finished Data preperation, look at code for more info* | Alejandro — Stage 1 data cleaning |
| `17634eb` | Sri Ramani Thungapati | 2026-04-04 | *EDA code added* | Ramani — Stage 1 exploratory data analysis |

---

## 4. How Stage 1 fed into Stage 2

The prepped CSV used by Himanshu's Stage 2 script
(`Dataset/prepped_hospital_data.csv`, produced by `Data_prep.py`) drops
identifiers and the leakage-prone `readmission_risk_score`, encodes `gender`
to 0/1, and keeps the variables that match the Stage 1 shortlist
(`age`, `comorbidities_count`, `length_of_stay`, `medications_count`,
`followup_visits_last_year`, `prev_readmissions`, plus the categorical fields
for `season`, `region`, `primary_diagnosis`, `treatment_type`,
`insurance_type`, `discharge_disposition`). That alignment is what lets the
Stage 2 baselines use a one-line column list per type without any further
feature engineering — the heavier engineering is Alejandro's Stage 2 task.

---

## 5. Branch / handoff notes

- This document and all Stage 2 artifacts are committed on the `stage2`
  branch only. `main` should not be modified by this work until the team
  agrees on the comparison.
- After review, merge `stage2` → `main` via PR (mirroring how Stage 1 was
  merged in `b4e68e6`).
- Stage 3 picks up from here: tune Logistic Regression and Gradient Boosting,
  calibrate the decision threshold, and lock the final model.
