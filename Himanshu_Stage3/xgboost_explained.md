# XGBoost vs Optimized Logistic Regression - Detailed Walkthrough

**Project:** 30-Day Hospital Readmission Risk Prediction (CSCI-485)
**Author:** Himanshu Singh Rao
**Date:** 6 May 2026
**Folder:** `Himanshu_Stage3/` (see [`README.md`](README.md) for layout)

This document explains exactly what XGBoost does, why it is a sensible
candidate for our hospital-readmission dataset, what every tuned
hyperparameter means in general and how the value we ended up with
reflects the data we have, and how the final tuned XGBoost compares
side-by-side with the Stage 3 optimized Logistic Regression. It closes
with a concrete answer to *can we squeeze more performance out of
XGBoost without overfitting?*.

The two models being compared:

- **Optimized Logistic Regression** - tuned with `GridSearchCV` in [`../Optimized_Logstic_model/Optimized_Logistic_model.py`](../Optimized_Logstic_model/Optimized_Logistic_model.py); results in [`../Optimized_Logstic_model/Logstic_Results.txt`](../Optimized_Logstic_model/Logstic_Results.txt).
- **Optimized XGBoost v1** - first tuning pass with `RandomizedSearchCV(n_iter=40)`; script [`v1_xgboost/train.py`](v1_xgboost/train.py); results in [`v1_xgboost/results.txt`](v1_xgboost/results.txt).
- **Optimized XGBoost v2** - wider search (`n_iter=100`) plus isotonic calibration; script [`v2_xgboost/train.py`](v2_xgboost/train.py); results in [`v2_xgboost/results.txt`](v2_xgboost/results.txt).
- **3-way head-to-head** in [`final_comparison/results.txt`](final_comparison/results.txt) and [`final_comparison/metrics.csv`](final_comparison/metrics.csv).

---

## 1. TL;DR comparison

| | **Logistic Regression (tuned)** | **XGBoost (tuned)** |
|---|---|---|
| Algorithm family | Linear (one weighted sum + sigmoid) | Gradient-boosted decision-tree ensemble |
| Hyperparameters tuned | 4 (C, l1_ratio, solver, class_weight) | 8 (n_estimators, learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, reg_lambda, gamma) + scale_pos_weight |
| Search method | `GridSearchCV`, 5-fold, scoring=`average_precision`, ~60 fits | `RandomizedSearchCV`, 5-fold, scoring=`average_precision`, 200 fits (40 candidates × 5 folds) |
| Class-imbalance handling | `class_weight="balanced"` | `scale_pos_weight = neg/pos = 0.2939` |
| Calibrated decision threshold | ~0.49 (F1-max on val PR curve - flags 59.25% of patients) | 0.2114 (F1-max on val PR curve - flags 89.33% of patients) |
| **Test ROC-AUC** | **0.8299** | **0.8252** |
| **Test PR-AUC** | **0.9371** | **0.9360** |
| Test Brier | 0.1706 | 0.1697 |
| Test F2 (β=2, recall-leaning) | 0.7658 | 0.9299 |
| Test confusion (TN / FP / FN / TP) | 221 / 52 / 268 / 659 | 90 / 183 / 38 / 889 |
| Patients flagged high-risk | 59.25% | 89.33% |

Both models land within ~0.005 ROC-AUC of each other - genuinely a tie on
ranking quality. The visible difference is *operating point*: the LR
threshold is much more conservative (flags 59% of patients), the XGBoost
threshold is much more aggressive (flags 89%). That explains the gulf in
F2 and confusion-matrix layout, **not** a real gap in model capability.

## 2. What XGBoost actually does

XGBoost (eXtreme Gradient Boosting) is an additive ensemble of small
**decision trees** trained sequentially. Each new tree is fit to the
*residual error* of the running ensemble. Mathematically the prediction
is

$$
\hat{y}_i = \sum_{k=1}^{K} f_k(x_i), \qquad f_k \in \mathcal{F}\ \text{(trees)}
$$

trained by greedily minimizing a regularized objective:

$$
\mathcal{L}(\phi) = \sum_i \ell(y_i, \hat{y}_i) \;+\; \sum_k \Omega(f_k), \qquad
\Omega(f) = \gamma\,T \;+\; \tfrac{1}{2}\lambda \|w\|^2
$$

where `T` is the number of leaves and `w` are the leaf weights. The
practically important things this gives us, and why it matters for
30-day readmission prediction on tabular hospital data:

| Property | What it means | Why it matters here |
|---|---|---|
| **Trees as base learners** | Each tree splits the feature space along axis-aligned thresholds. | Captures non-linear effects (e.g. risk rises sharply only above LOS > 7 days) and interactions (e.g. age × insurance_type) without any feature engineering. |
| **Sequential additive fitting** | Each tree corrects the residual of the previous ensemble. | Lets the model represent complex risk surfaces with many *small* trees rather than one big over-fit tree. |
| **Second-order Taylor expansion** | XGBoost uses both the gradient *and* Hessian of the loss when scoring splits. | Faster, more stable convergence than first-order boosters (e.g. classical AdaBoost). |
| **Built-in regularization** (`λ`, `γ`) | Penalizes leaf weights and tree size in the objective itself, not as a post-hoc step. | This is the single biggest lever against overfitting on small clinical datasets like ours (5,600 training rows). |
| **Histogram-based splits** (`tree_method="hist"`) | Bins each feature into ≤ 256 buckets before searching for splits. | Massive speedup on numeric columns like `age` and `length_of_stay`; the model we report uses `tree_method="hist"`. |
| **Missing-value-aware splits** | At each split, missing values are routed to whichever direction reduces loss the most, learned per-split. | Even though our prepped CSV has no NaNs, this means raw EHR-style data with missing labs would Just Work without imputation. |
| **`scale_pos_weight`** | Multiplies the gradient of positive-class examples by a constant. | The standard XGBoost hook for our 77/23 class imbalance - we use `neg/pos = 0.2939` to *down*-weight the majority positives. |
| **`predict_proba` is well-defined** | Output of the final logistic transform is a calibratable probability. | We can apply the same F1-max threshold-calibration trick we used for LR, and score with PR-AUC and Brier. |

In short: it is a strictly more flexible function class than logistic
regression, with regularization built in. Whether that flexibility
*helps* depends on whether the dataset has signal that a linear model
can't capture. (On our prepped data, it almost doesn't - section 4
explains why.)

## 3. Why XGBoost is a reasonable fit for *this* dataset

The Stage 1 variable summary
([`stage1_variable_summary.md`](../stage1_variable_summary.md)) and the
Stage 2 literature review
([`Himanshu_Stage2/literature_review.md`](../Himanshu_Stage2/literature_review.md))
together pointed at six numeric utilization/clinical predictors
(`age`, `comorbidities_count`, `length_of_stay`, `medications_count`,
`followup_visits_last_year`, `prev_readmissions`) and six categorical
ones (`season`, `region`, `primary_diagnosis`, `treatment_type`,
`insurance_type`, `discharge_disposition`) plus a binary `gender`.

XGBoost is well-matched for this shape of data because:

1. **Mixed numeric + one-hot categorical input is its natural diet.** The same `Pipeline(OneHotEncoder + StandardScaler)` from Stage 2 / LR feeds straight into it - no extra feature engineering required.
2. **Several Stage 1 predictors are non-monotonic in the literature.** Length of stay, in particular, is *both* a "very-short discharge" and "very-long stay" risk - U-shaped in ROC space. A single linear coefficient cannot represent that; a depth-5 tree can.
3. **Interactions are clinically expected.** E.g. `prev_readmissions × discharge_disposition` (Skilled Nursing) and `age × comorbidities_count`. Trees model interactions implicitly.
4. **The literature review puts XGBoost in the top 2-3 single-model results** on every readmission paper that runs it (Mubarak 2025 0.667; Liu et al. 0.64; Li 2024 ~0.70; Hidayaturrohman & Hanada 2024 top-tier; Sumon et al. 2025 ensemble component). It is the most widely benchmarked GBDT for this task.

That said, two limitations specific to *our* data temper the upside:

- **Only ~5,600 training rows.** Boosted trees benefit most from large data; below ~50k rows the marginal improvement over a well-regularized LR is small.
- **Synthetic dataset.** Our CSV looks synthetic-Kaggle-style (literature review §1) - the categorical-numeric interactions a tree could exploit may simply not be there. Reading the headline numbers, the four Stage 2 baselines clustered within 0.03 ROC-AUC of each other, which is exactly the *signal-ceiling* signature.

That tells us up front: a tuned XGBoost is unlikely to beat LR by a wide
margin, and any tuning gains will be in the second decimal of ROC-AUC,
not the first.

## 4. How we optimized the XGBoost hyperparameters

Source: [`optimized_xgboost.py`](optimized_xgboost.py). All choices are
reproducible with `random_state=42`.

### 4.1 Cross-validation strategy

- **`StratifiedKFold(n_splits=5, shuffle=True)`** on the 5,600-row training split. Stratified so each fold preserves the 77/23 positive-rate ratio.
- **Validation split (1,200 rows) is held out** during the search and used only for threshold calibration after the best estimator is refit. The 1,200-row test split is touched **once** at the end.
- **Scoring metric: `average_precision` (PR-AUC).** Picked because the dataset is imbalanced *toward* positives - PR-AUC reflects the operating regime we care about (precision/recall on readmissions) much better than accuracy or even ROC-AUC.

### 4.2 Search method: `RandomizedSearchCV`, not GridSearchCV

| | Grid | Random |
|---|---|---|
| Cost for an 8-D space | exponential | linear in `n_iter` |
| Coverage of important dims | uniform on the chosen grid | denser on continuous dims, with no wasted compute on unimportant ones |
| Reproducibility | exact | exact (with a `random_state`) |

For 8 hyperparameters, a sensible grid would be 4-5 values per axis →
6⁵ ≈ 7,776 candidates × 5 folds = ~39,000 fits. Randomized search with
`n_iter=40 × 5 folds = 200 fits` covers the same volume statistically
much better than that grid would (Bergstra & Bengio 2012, often cited
in the medical-ML literature in our review).

For LR we kept `GridSearchCV` because it has only 4 hyperparameters and
6 × 5 × 1 × 2 = 60 candidates is cheap to enumerate exhaustively.

### 4.3 Why we don't use early stopping inside the CV

XGBoost can early-stop on a validation set inside each fit. We did
*not* wire that into `RandomizedSearchCV` because the cleaner pattern
is: (1) tune `n_estimators` as part of the search itself, and (2) keep
our 1,200-row val split for *threshold* calibration only, not for
early-stop-fold-leakage. This matches the pattern in
[`Optimized_Logstic_model.py`](../Optimized_Logstic_model/Optimized_Logistic_model.py)
and avoids the val-set being used twice for two different decisions.

### 4.4 Threshold calibration

After the best estimator is refit, we score the validation set, sweep
the precision-recall curve, and pick the threshold that maximizes F1.
This is the same recipe used by `Optimized_Logistic_model.py`. The F1
maximizer chosen here is `0.2114` - aggressive, because the imbalance
is *toward* the positive class.

## 5. Each hyperparameter, in detail

The table that follows covers every knob we tuned. For each row:

- **Effect** = what the parameter does to bias / variance.
- **Direction** = which way to move it if the model is overfitting / underfitting.
- **Search range** = the distribution we sampled from.
- **Final value** = the value picked from the validation-set best score.
- **What that value tells us about *our* data.**

| Hyperparameter | What it does | Bias-variance effect | Search range | Final value | What this value says about our data |
|---|---|---|---|---|---|
| **`n_estimators`** | Number of boosting rounds = number of trees. | More trees = more capacity. Past a point, you fit noise. | `randint(200, 900)` | **574** | Mid-range. Not maxed out - the search is happy with a moderate ensemble, consistent with a small dataset where extra trees don't earn their keep. |
| **`learning_rate` (η)** | Multiplier shrinking the contribution of each new tree. | Lower η = slower learning = needs more trees but generalizes better. | `loguniform(0.01, 0.3)` | **0.0138** | At the *very low* end of the range. The optimizer is leaning hard on slow learning - a textbook anti-overfitting move on small data. Combined with 574 trees it is essentially "many slow corrections" instead of "few aggressive ones". |
| **`max_depth`** | Max depth of each tree. Higher = bigger interactions modeled per tree. | Higher = lower bias / higher variance. | `randint(3, 9)` | **5** | Mid-depth. The data supports 4-way interactions but the search did not push to depth 8 - probably because the synthetic feature set doesn't contain meaningful 5+ way interactions. |
| **`min_child_weight`** | Minimum sum of instance Hessians required in a child node before a split is allowed. Acts like a minimum-leaf-size guard. | Higher = simpler trees = less overfitting. | `randint(1, 10)` | **6** | Notably high. This is a strong leaf-size constraint, telling us the search prefers chunky, robust leaves. Sensible on 5,600 rows where deep splits could be cutting noise. |
| **`subsample`** | Fraction of training rows sampled (without replacement) for each tree. Stochastic-gradient flavor. | Lower = noisier per-tree fits = decorrelates the ensemble. | `uniform(0.6, 1.0)` | **0.836** | Mild row subsampling. Each tree sees ~83% of the rows. Gives some regularization, but the search didn't aggressively subsample - again consistent with "we don't have a ton of rows to begin with". |
| **`colsample_bytree`** | Fraction of features sampled at each tree. Decorrelates trees by forcing them to use different views of the data. | Lower = more decorrelation = stronger ensemble averaging. | `uniform(0.6, 1.0)` | **0.863** | Mild column subsampling. Most features are kept per tree - sensible because we only have ~30 columns post-one-hot, so dropping them aggressively would actually starve the trees. |
| **`reg_lambda` (L2)** | L2 penalty on leaf weights. Exact analog of LR's `C`. | Higher = smaller leaf values = smoother predictions. | `loguniform(0.01, 10)` | **0.0564** | Surprisingly *low*. The search wanted very little L2. That, combined with high `gamma` (next row), means the optimizer prefers to control complexity by *not splitting* rather than by shrinking leaf values. |
| **`gamma`** | Minimum loss reduction required for a split to be retained. Acts like a complexity tax per split. | Higher = fewer, only-if-clearly-useful splits. | `uniform(0, 5)` | **2.842** | High - on the upper half of the search range. Clear signal: the search prefers conservative trees that only split when the gain is genuinely large. This is consistent with a small dataset where weak splits are likely noise. |
| **`scale_pos_weight`** *(fixed, not searched)* | Multiplies positive-class gradients. The XGBoost analog of `class_weight="balanced"`. | Brings the loss back to balanced even when the data isn't. | fixed `neg/pos = 0.2939` | **0.2939** | Computed from the training distribution. Pulls the optimizer toward attending to the *minority negative class* - the one we'd otherwise miss. |
| **`tree_method`** *(fixed)* | Split-finding algorithm. | `hist` is faster + uses less memory. | fixed `"hist"` | **`"hist"`** | Standard recommendation for 4 KB-1 MB tabular data. |
| **`eval_metric`** *(fixed)* | Internal metric XGBoost monitors during training. | Doesn't affect tree fits, only what gets logged. | fixed `"logloss"` | **`"logloss"`** | The matching probabilistic loss. |

### Reading the *combination* of values

Almost every chosen value points the same direction: **regularize hard,
trust the data lightly**.

- `learning_rate` 0.014 (very low)
- `gamma` 2.84 (high)
- `min_child_weight` 6 (high)
- `n_estimators` 574 (mid - many slow corrections)
- `max_depth` 5 (mid)

That is the canonical "small-dataset XGBoost" profile. The
hyperparameter search has effectively done the bias-variance trade-off
itself, and it has landed firmly on the safe side - which is the
correct behavior given that we only have 5,600 training rows of
synthetic data.

## 6. Logistic Regression's tuned settings, for the head-to-head

For completeness:

| Hyperparameter | Final value | What it means |
|---|---|---|
| `C` | `0.1` | Strong L2 regularization. Smaller `C` = more regularization. |
| `solver` | `saga` | Required to support `elasticnet`. |
| `l1_ratio` | `0` | Pure L2 (despite the elastic-net solver being available). |
| `class_weight` | `balanced` | Up-weights the minority negative class to balance the loss. |
| `max_iter` | `2000` | Convergence headroom for `saga`. |
| Calibrated threshold | ~`0.49` | F1-max on the val PR curve (flags 59% of patients on test). |

LR's tuning verdict is also *regularize hard*: `C=0.1` is a 10× tighter
penalty than the default. Both models tell the same story about the
data: small, partly synthetic, and most of the signal lives in a
handful of monotonic predictors.

## 7. LR vs XGBoost - head-to-head

### 7.1 Where they actually differ

| Dimension | LR | XGBoost |
|---|---|---|
| Decision boundary | One hyperplane in feature space | Sum of axis-aligned partitions; can carve any boundary the data supports |
| Feature interactions | None unless we pre-multiply features | Implicit, up to depth `max_depth` per tree |
| Non-monotonic effects | Cannot model (one coefficient per feature) | Modeled by definition |
| Robustness to scaling / outliers | Sensitive; we use `StandardScaler` | Insensitive (splits are rank-based) |
| Robustness to missing values | Requires imputation | Native missing-direction learning |
| Interpretability | Direct - one coefficient per feature, with odds-ratio | Per-tree gain or SHAP; less direct but richer |
| Compute | Sub-second per fit | ~5-10 s per fit on this data; full search ~3 min |
| What overfitting looks like | Coefficients blow up; cured by lower `C` | Trees memorize folds; cured by lower `learning_rate`, higher `gamma`, `min_child_weight` |

### 7.2 Why LR is competitive on *this* data

Three reasons, all evidenced in the numbers:

1. **The signal is mostly monotonic.** `prev_readmissions ↑ → risk ↑`, `comorbidities_count ↑ → risk ↑`, etc. These are exactly the features XGBoost's gain ranking puts at the top - and a linear model can represent monotonic effects perfectly.
2. **The dataset is small.** 5,600 training rows is well-trodden LR territory. XGBoost's "extra capacity" only helps when we feed it tens of thousands of rows.
3. **The categoricals are low-cardinality.** With at most ~5 levels per category, one-hot expansion gives LR a fully-served representation - it does not need a tree to "find" the right level interactions.

The flip side of the same observation: if we had real EHR data with
50k+ rows, more numeric labs, and high-cardinality codes (ICD-10,
medications), the gap would open up - this is exactly what TabArena
2025 and Mubarak 2025 report on the larger UCI Diabetes 130-US
benchmark in the Stage 2 literature review.

### 7.3 What changed between XGBoost v1 and v2

`v2_xgboost/` is the answer to "can we tune further without
overfitting?" - a wider random search (`n_iter` 40 → 100), three
extra hyperparameters (`reg_alpha`, `colsample_bylevel`,
`max_delta_step`), an extended `learning_rate` floor (down to 0.005),
and `n_estimators` ceiling (up to 1500), plus
`CalibratedClassifierCV(method="isotonic", cv=5)` wrapped around the
tuned model.

Test-set delta v1 → v2 (full numbers in
[`final_comparison/results.txt`](final_comparison/results.txt)):

| Metric | v1 | v2 | Delta |
|---|---:|---:|---:|
| ROC-AUC | 0.8252 | 0.8266 | +0.0014 |
| PR-AUC | 0.9360 | 0.9373 | +0.0014 |
| **Brier** | 0.1697 | **0.1304** | **−0.0393** (lower = better) |
| F2 | 0.9299 | 0.9363 | +0.0064 |

ROC-AUC and PR-AUC each gained ~0.001 - at the noise floor of a
1,200-row test split. The real win is **calibration**: Brier dropped
23%, which moves v2's reliability curve onto the diagonal in
[`final_comparison/figures/calibration.png`](final_comparison/figures/calibration.png).

## 8. Can we tune XGBoost further without overfitting? - practical answer

### 8.1 What the v2 numbers tell us

The v1 five-fold CV PR-AUC was **0.9349**, validation **0.9433**, test
**0.9360**. The v2 numbers are **0.9357 / 0.9463 / 0.9373**. Reading
that out:

- Train-CV-test gap is essentially flat in both versions. There is no
  visible overfitting to the training data even after a much wider
  search.
- The gap between *validation* and *test* is small but real - that is
  the threshold/refit overhead, expected.
- v2 confirms the headroom *exists* but is small. Tuning + calibration
  bought ~0.001 PR-AUC and a 23% Brier improvement.

The question is no longer "can we tune further" - we did. The question
is "what would buy more than another 0.001 of PR-AUC."

### 8.2 What further tuning would buy

| Lever | Expected upside on this data | Overfitting risk |
|---|---|---|
| Wider random search (`n_iter` = 100-200) | Likely a few thousandths of PR-AUC. Tiny. | None - the search itself doesn't overfit if the CV is honest. |
| Bayesian search (Optuna) instead of random | Same upside as a wider search, faster to converge. | Same as above. |
| Lower `learning_rate` (0.005) + higher `n_estimators` (~1500) + early stopping | A few thousandths. Cleanly trades compute for stability. | Low, *if* early stopping uses an inner CV fold not the val split. |
| More features (Alejandro's Stage 2 feature engineering: prior-ED-visit ratios, drug-class buckets, …) | This is where the real upside is. Real new signal can lift both LR and XGBoost. | Moderate - new features need to be vetted for leakage and added to *both* train and validate scoring pipelines. |
| Stacking LR + XGBoost behind a logistic meta-learner | Usually a few thousandths over the best base model. The two head models already disagree only on the operating point, not the ranking - limited stacking upside on this data. | Low if the meta-learner is fit on *out-of-fold* base predictions, not in-fold. |
| Going deeper / more trees without compensating regularization | Negligible upside on this size of data. | **High** - this is the classic XGBoost overfitting failure mode. |

The pragmatic answer, now confirmed by the v2 experiment:

> **Yes - slightly on ranking, substantially on calibration.** The
> wider random search (`n_iter=40 → 100`) + isotonic calibration in
> `v2_xgboost/` added **+0.0014 ROC-AUC, +0.0014 PR-AUC, and a 23%
> Brier improvement** with no sign of overfitting (train-CV-test gap
> stayed flat). Anything bigger than that single-decimal-millis ranking
> gain needs **new features**, not more tuning.

### 8.3 What v3 would look like, if we wanted to push further

v2 already implements most of the recipe (`reg_alpha`,
`colsample_bylevel`, `max_delta_step`, isotonic calibration). The
remaining levers, none of which are cheap or guaranteed:

```python
# 1. Replace RandomizedSearchCV with an Optuna study that maximizes the
#    out-of-fold PR-AUC (5-fold StratifiedKFold), with early stopping
#    inside each fold using a held-out 10% of the *fold's* training rows.
# 2. Same v2 search ranges, but doubled n_iter (~200) -- Optuna's TPE
#    will spend that on promising regions instead of uniform sampling.
# 3. Add Platt scaling alongside isotonic and pick whichever calibrator
#    gives the lower validation Brier.
```

Two guardrails to keep this honest:

1. **Don't peek at the test set during the new search.** The 1,200-row test split has been touched once - keep it that way until the new model is locked.
2. **Compare *test-set Brier* in addition to ROC-AUC / PR-AUC.** Calibration is the metric that breaks first when XGBoost is over-tuned; it's a leading indicator that the new model is overfitting before AUC moves.

## 9. Bottom line

- **XGBoost is well-suited to this kind of data, but not uniquely so.** On our 5,600-row, mostly-monotonic, mostly-low-cardinality feature set, a regularized LR is genuinely competitive. The Stage 3 final-comparison test ROC-AUC ranking is LR > XGBoost v2 > XGBoost v1 - all within ~0.005 of each other.
- **The tuned XGBoost in `v1_xgboost/train.py` is already inside the family of "well-regularized small-data XGBoost" models.** Every chosen hyperparameter points at conservative, regularized fitting.
- **`v2_xgboost/train.py` is the answer to "can we tune further without overfitting?"** Wider search + isotonic calibration moved every metric in the right direction (Brier dropped 23%) with no sign of overfitting. Further ranking gains beyond v2 will require *new features* (Alejandro's Stage 2 engineering work), not more knob-turning.
- **For the team's Stage 4 interpretability work, XGBoost v1's gain ranking** (`comorbidities_count`, `insurance_type`, `medications_count`, `age`, `prev_readmissions`) lines up directly with the Stage 1 literature shortlist - which is the cleanest sanity check we can produce that the right predictors are doing the work.
