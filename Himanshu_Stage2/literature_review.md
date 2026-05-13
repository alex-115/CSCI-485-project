# Stage 2 — Literature Review: What Other People Have Done

**Author:** Himanshu Singh Rao
**Date:** 25 April 2026
**Purpose:** Survey what models the published literature uses for 30-day
hospital readmission prediction, what's been done specifically with our
dataset (or its closest analogs), and recommend additional candidates the
team should consider for Stages 3–4.

---

## 1. Our dataset in context

`Dataset/Hospital_dataset.csv` (8,000 patients, 16 features +
`readmission_risk_score` + `label`) does **not** match the canonical UCI
Diabetes 130-US Hospitals dataset that Stage 1's
`stage1_variable_summary_full.md` originally targeted. Its column signature
(`season`, `region`, `comorbidities_count`, `length_of_stay`,
`medications_count`, `followup_visits_last_year`, `prev_readmissions`,
`insurance_type`, `discharge_disposition`, plus a leakage-prone synthetic
`readmission_risk_score`) matches the **synthetic Kaggle "Hospital
Readmission Prediction"-style** datasets — for example
`vanpatangan/readmission-dataset` and `dubradave/hospital-readmissions`.

Implication: published numbers are mostly reported on either the **UCI
Diabetes 130-US** dataset (the academic standard) or the synthetic Kaggle
datasets. The two regimes are very different:

| Setting | Typical AUROC reported | Why |
|---|---|---|
| UCI Diabetes 130-US (101k rows, real EHR) | **0.60 – 0.77** | Genuinely hard signal; ~11% positive class. |
| Synthetic Kaggle readmission datasets (~8–25k rows) | **0.83 – 0.95** | Engineered features and synthetic labels make the problem easier. |
| Our project run (8k rows, synthetic) | **0.83 – 0.86** (this branch's notebook) | Sits right in the synthetic-Kaggle regime — consistent with what others get. |

So our headline numbers (LR 0.858 / XGBoost 0.837) are entirely in line with
what the synthetic-dataset literature reports. The interesting question is
**which models other people use** and **whether any of them are likely to
push us higher in Stage 3.**

## 2. Models other people use for 30-day readmission

Below is a synthesis of recent (2021–2025) papers. For each model I list
typical reported performance and whether it's worth carrying into Stage 3.

| Family | Model | Used in (sample papers) | Typical AUROC | Strengths | Weaknesses | Verdict for our project |
|---|---|---|---|---|---|---|
| Linear | **Logistic Regression** | Almost every paper as a baseline; Cornell Data Journal 2022 [^cornell]; Mubarak 2025 [^mubarak]; Liu et al. JMAI [^liu] | 0.58 – 0.77 | Simple, interpretable, fast, easy to defend in Stage 4. | Cannot capture non-linear interactions. | **Already in our notebook.** Keep as the interpretable baseline. |
| Tree | **Random Forest** | Same as above + brunoarine GitHub [^bruno] | 0.63 – 0.73 | Decent off the shelf; little tuning needed. | Often middle-of-the-pack; poor calibration. | **Already in our notebook.** Keep but don't carry as a finalist. |
| Boosting | **XGBoost** | Mubarak 2025 [^mubarak]; Shakil et al. 2024 [^shakil]; Li 2024 [^li]; Hidayaturrohman & Hanada 2024; AHF 30-day study (PMC) [^ahf]; INCOFT 2025 [^incoft] | 0.64 – 0.86 | Most-cited best classical model. Strong default; rich tuning surface; built-in regularization, class-weighting via `scale_pos_weight`. | Many knobs to tune. | **Already in our notebook.** Keep; tune in Stage 3. |
| Boosting | **LightGBM** | AHF 30-day study [^ahf]; Gandra 2024 [^gandra]; numerous Kaggle solutions; TabArena 2025 [^tabarena] | 0.65 – 0.83 | Faster than XGBoost on big data; native categorical support; on TabArena's leaderboard it's a top-3 single model after post-hoc ensembling. | Slightly less robust on small data than CatBoost. | **Worth adding.** Cheap to include; close cousin of XGBoost. |
| Boosting | **CatBoost** | Gandra 2024 (best on UCI, AUC 0.70) [^gandra]; Frail-readmission study (PMC, AUC 0.79) [^frail]; AHF 30-day study [^ahf]; INCOFT 2025 (AUC 0.94, top single model) [^incoft]; TabArena 2025 (#1 single model under conventional tuning) [^tabarena] | 0.70 – 0.94 | **The most consistently top-performing single model in recent readmission papers.** Native handling of categorical columns (no one-hot needed), great defaults, strong out-of-the-box performance. | New dependency; slightly slower than LightGBM. | **Strongly recommended.** Adds the most upside vs. effort. |
| Deep | **MLP / DNN** | Mubarak 2025 [^mubarak]; Liu et al. [^liu]; INCOFT 2025 [^incoft] | 0.58 – 0.83 | Can capture non-linear interactions. | Often *underperforms* GBDT on tabular data (well-documented in Yildiz 2024 [^gbdt-medical]); needs more data and tuning. | **Not recommended** as a primary model — listed mainly as a deep-learning point of comparison. |
| Deep (tabular-specific) | **TabNet** | Frontiers ED-revisit study 2025 (AUC 0.867) [^tabnet-er]; CL pre-trained EHR work 2025 (AUC 0.81) [^tabnet-cl] | 0.76 – 0.87 | Built-in feature attention; competitive with GBDT on some EHR tasks. | Heavier deps (PyTorch), longer to train, more sensitive to hyperparameters. | **Optional stretch.** Only worth it if Stage 3 has time after tuning the GBDTs. |
| Deep (tabular-specific) | **TabTransformer / TabPFN(v2.5) / TabM / RealMLP** | TabArena 2025 living benchmark [^tabarena]; TabPFN-2.5 tech report 2025 [^tabpfn] | varies | TabPFN-2.5 *zero-shot* now outperforms tuned XGBoost/CatBoost on small datasets ≤ 50k rows — directly relevant to our 8k-row regime. | Requires PyTorch + GPU for best speed; foundation model. | **Stretch experiment** for Stage 3 if anyone has time and a GPU. Not core. |
| Ensemble | **Weighted ensemble of CatBoost + XGBoost + MLP** | Sumon et al., INCOFT 2025 [^incoft] (87.08% accuracy, AUC ≈ 0.94) | 0.85 – 0.95 | Currently SOTA on multiple synthetic-Kaggle readmission datasets. | Less interpretable; multiple models to maintain. | **Recommended Stage 3 stretch** once individual models are tuned. |

## 3. What people have done specifically with the UCI Diabetes 130-US dataset

The closest published benchmark — and the one the Stage 1 doc originally
targeted. Numbers are notably lower than ours because the UCI task is
genuinely harder and uses only structured EHR features.

| Source | Models | Best AUROC reported |
|---|---|---|
| **Strack et al. 2014** (original UCI paper) [^strack] | Logistic regression, decision tree analyses | ~0.62 |
| **Cornell Data Journal 2022** [^cornell] | Logistic regression | 0.77 |
| **Mubarak et al. 2025 (PMC)** [^mubarak] | LR, RF, XGBoost, DNN | XGBoost 0.667 |
| **Liu et al., J. Medical AI 2024** [^liu] | LR, RF, XGBoost, MLP, DT, SVM, KNN, AdaBoost, NB, LSTM | XGBoost 0.64; RF 0.63; LSTM 0.61 |
| **Gandra 2024** [^gandra] | LR, RF, GB, XGBoost, LightGBM, **CatBoost** | **CatBoost 0.70** (top); LR 0.65; RF 0.64 |
| **brunoarine GitHub 2021** [^bruno] | LR, RF | LR 0.64; RF 0.66 |
| **Li 2024** [^li] | LR, RF, GBDT, DT, **XGBoost**, DNN | XGBoost ~0.70 |

**Key takeaway:** on UCI, the field has converged on a fairly tight cluster
where **CatBoost or XGBoost are typically the best single classical models
(AUROC 0.66 – 0.70)** and deep learning has not consistently won.

## 4. What people have done with synthetic Kaggle readmission datasets (closest to ours)

| Source | Dataset family | Best model | Best metric |
|---|---|---|---|
| **Shakil et al. 2024 (JCSTS)** [^shakil] | Patient demographic + clinical synthetic dataset | XGBoost | "highest performance" (paper reports ROC-AUC qualitatively) |
| **Sumon et al., INCOFT 2025** [^incoft] | Hospital-readmission Kaggle-style dataset | **Ensemble (CatBoost + XGBoost + MLP, weights 0.35/0.35/0.20)** | Acc 87.08%, AUC ≈ 0.94 |
| **Hidayaturrohman & Hanada 2024** | Synthetic readmission set | XGBoost | top-tier reported |

Our notebook's val AUROC 0.83 – 0.86 is consistent with the **lower end** of
this regime. The ensemble approach in Sumon et al. (CatBoost weighted
heavier than XGBoost) is the most credible single recipe to try.

## 5. Recurring "important features" across studies

Almost every paper above lists the same handful of top features driving the
prediction. **All of these are already in our prepped CSV** — which is good
news for our Stage 3 work and consistent with `stage1_variable_summary.md`.

| Feature | Prominence |
|---|---|
| Prior inpatient / readmission count (`prev_readmissions`) | Top-3 in [^frail] [^cornell] [^mubarak] [^shakil] [^bruno] |
| Length of stay (`length_of_stay`) | Top-3 in [^cornell] [^shakil] [^ahf] [^bruno] |
| Number of medications (`medications_count`) | Top-3 in [^gandra] [^mubarak] [^bruno] |
| Comorbidity count / number of diagnoses (`comorbidities_count`) | Top-5 in [^frail] [^mubarak] |
| Age (`age`) | Top-5 in [^cornell] [^bruno] [^ahf] |
| Discharge disposition (`discharge_disposition`) | Top-3 in [^frail] [^bruno] |

The XGBoost feature-importance plot in our notebook
(`figures/xgboost_top_features.png`) shows the same predictors at the top —
which is a healthy sanity check.

## 6. Recommended additions for Stage 3

Based purely on what's most likely to *actually move our metrics*:

1. **Add CatBoost to the candidate set.** It has the best track record as a
   single model on this kind of problem, handles our raw categorical
   columns natively without one-hot expansion, and adds only one
   dependency. Expected upside: 0.5 – 1.5 ROC-AUC points over XGBoost on
   our data.
2. **Add LightGBM** as a cheap third tree-based alternative; sometimes
   wins on speed and never far behind XGBoost.
3. **Build a Stage 3 weighted ensemble** of class-weighted Logistic
   Regression + tuned XGBoost + tuned CatBoost following the Sumon et al.
   recipe (their best weights were 0.35 / 0.35 / 0.20). This is the
   credible path to 0.90+ ROC-AUC if it's reachable on this data.
4. **Use SHAP for Stage 4 interpretability.** Every recent paper uses it;
   it gives both global and per-patient explanations and is the standard
   for "which inputs the model weights most" — exactly what Himanshu's
   Stage 4 task asks for.
5. **Use stratified k-fold CV (k = 5) + Bayesian / random hyperparameter
   search** for tuning, mirroring [^ahf] and [^liu]. Avoid grid search on
   wide spaces.
6. **Consider class-weighting *or* SMOTE** during training (most papers
   use one or the other). Our data is imbalanced *toward* the positive
   class (77%), so we should weight the **negative** class up — which is
   already what `class_weight="balanced"` does in our LR.
7. **Stretch (only if time permits in Stage 3):** TabPFN-2.5. The
   TabArena and TabPFN-2.5 tech-report results suggest it can outperform
   tuned tree-based models *zero-shot* on datasets of our size (~8k rows).
   It's a single-line drop-in if PyTorch is acceptable.

### What I would *not* recommend

- **Plain MLP / DNN.** Recent benchmarks (TabArena 2025 [^tabarena];
  Yildiz 2024 [^gbdt-medical]) confirm that without ensembling and heavy
  tuning, MLPs lose to GBDTs on tabular medical data. Not worth the time.
- **LSTM.** Liu et al. [^liu] showed LSTM at 0.61 ROC-AUC on UCI — strictly
  worse than XGBoost. Our data has no temporal sequence per patient
  anyway, so LSTM has no structural advantage.
- **Multimodal / clinical-notes models** (PT, EHR-text CL [^pt] [^ehr-cl]).
  Genuinely state-of-the-art but require unstructured text we don't have.

## 7. Summary table — what to carry into Stage 3

| Model | Currently in notebook? | Carry into Stage 3? | Why |
|---|---|---|---|
| Logistic Regression (class-weighted) | Yes | **Yes** | Current leader on our data; interpretable. |
| Random Forest | Yes | No | Default-RF is dominated by GBDTs. |
| Gradient Boosting | Yes | No | Subsumed by XGBoost. |
| XGBoost | Yes | **Yes** | Largest tuning surface among GBDTs. |
| **CatBoost** | No | **Yes (add)** | Most consistent top performer in recent papers. |
| **LightGBM** | No | Yes (cheap add) | Fast tree alternative; native categoricals. |
| **Ensemble (LR + XGBoost + CatBoost)** | No | **Yes (Stage 3 stretch)** | Sumon et al. 2025 ensemble recipe. |
| TabPFN-2.5 | No | Optional | Foundation model; zero-shot SOTA on small data. |
| MLP / DNN | No | No | Loses to GBDTs on tabular. |
| TabNet / TabTransformer | No | Optional stretch | Heavier deps; uneven returns. |
| LSTM / multimodal | No | No | Wrong data shape. |

---

## 8. References

[^cornell]: Cornell Data Journal, *Predicting Diabetes Readmissions in Hospitalized Patients Using Machine Learning* (UCI Diabetes 130-US, logistic regression, AUC 0.77). https://cornelldatajournal.org/articles/diabetes-readmission

[^mubarak]: Mubarak et al. (2025). *Predicting 30-Day Hospital Readmission in Patients With Diabetes Using Machine Learning on Electronic Health Record Data.* PMC12085305. UCI Diabetes 130-US; LR 0.642 / RF 0.630 / XGBoost 0.667 / DNN 0.579. https://pmc.ncbi.nlm.nih.gov/articles/PMC12085305/

[^liu]: Liu et al. *Comparison of machine learning models for predicting 30-day readmission rates for patients with diabetes.* Journal of Medical Artificial Intelligence. 11 ML models including XGBoost (0.64 AUROC), RF (0.63), LSTM (0.61). https://jmai.amegroups.org/article/view/9179/html

[^bruno]: brunoarine, *diabetes* GitHub repo. UCI Diabetes 130-US: LR AUC 0.64, RF AUC 0.66. https://github.com/brunoarine/diabetes

[^gandra]: Gandra A. (2024). *Predicting Hospital Readmissions in Diabetes Patients: A Comparative Study of Machine Learning Models.* International Journal of Health Sciences. UCI Diabetes 130-US; CatBoost AUC 0.70 (best of 6 models).

[^shakil]: Shakil M.R. et al. (2024). *Machine Learning-Based Hospital Readmission Prediction and Risk Analysis in the United States Healthcare System.* J. Computer Science and Technology Studies, 6(5), 369-384. XGBoost top-performer on demographic + clinical synthetic dataset. https://al-kindipublishers.org/index.php/jcsts/article/view/12431

[^li]: Li C. (2024). *Machine learning-based readmission risk prediction for diabetic patients.* Applied and Computational Engineering, 46, 45-59. XGBoost top of 6 classifiers (incl. DNN). https://www.ewadirect.com/proceedings/ace/article/view/10780

[^ahf]: Anonymous (2024). *Explainable machine learning for predicting 30-day readmission in acute heart failure patients.* PMC11261142. XGBoost AUC 0.763 > CatBoost 0.710 > LightGBM 0.729 > RF 0.726 > LR 0.584. https://pmc.ncbi.nlm.nih.gov/articles/PMC11261142/

[^frail]: *Machine learning for predicting readmission risk among the frail.* PMC8767300. CatBoost best of 5 models, AUC 0.79. https://pmc.ncbi.nlm.nih.gov/articles/PMC8767300/

[^incoft]: Sumon et al. (2025). *Hospital Readmission Prediction using Weighted Ensemble of CatBoost, XGBoost and MLP.* INCOFT 2025. Ensemble accuracy 87.08%, AUC ≈ 0.94. https://www.scitepress.org/Papers/2025/136032/

[^strack]: Strack B., DeShazo J. P., Gennings C., et al. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records.* BioMed Research International, vol. 2014, Article ID 781670.

[^tabarena]: Erickson N. et al. (2025). *TabArena: A Living Benchmark for Machine Learning on Tabular Data.* NeurIPS 2025. CatBoost #1 single model under conventional tuning; deep models catch up only with post-hoc ensembling. https://arxiv.org/abs/2506.16791

[^tabpfn]: Prior Labs (2025). *TabPFN-2.5 technical report.* Foundation model; zero-shot beats tuned XGBoost / CatBoost on TabArena-Lite up to 50k rows.

[^gbdt-medical]: Yildiz A.Y. (2024). *Gradient Boosting Decision Trees on Medical Diagnosis over Tabular Data.* arXiv:2410.03705. GBDTs (XGBoost / CatBoost / LightGBM) outperform DL methods (TabNet, TabTransformer) on tabular medical data with much lower compute.

[^tabnet-er]: Frontiers in Public Health (2025). *Performance comparison of artificial intelligence models in predicting 72-h ED unscheduled return visits.* TabNet AUROC 0.867 (best of 5 models). https://www.frontiersin.org/journals/public-health/articles/10.3389/fpubh.2025.1609206

[^tabnet-cl]: arXiv:2505.17643 (2025). *Multimodal contrastive learning for EHR + discharge notes.* Pretrained TabNet improves 30-day readmission AUC by ~4% over XGBoost.

[^pt]: arXiv:2412.12909 (2024). *Plain Transformer for multimodal 30-day readmission* (EHR + chest X-ray + clinical notes).

[^ehr-cl]: arXiv:2505.17643. (Same as [^tabnet-cl].)
