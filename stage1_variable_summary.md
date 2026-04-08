# Stage 1 — Variables & Definitions for 30-Day Readmission Prediction

**Author:** Himanshu Singh Rao
**Date:** 7 April 2026

Short summary of the clinical, demographic, and utilization predictors worth prioritizing in our data, based on the readmission literature.

---

## Demographic Variables

| Variable | Definition |
|----------|-----------|
| Age | Patient age (often bucketed in 10-year intervals). Older patients have consistently higher readmission risk. |
| Gender | Male / Female / Other. |
| Race / Ethnicity | Self-reported category. Relevant for capturing access and outcome disparities. |

## Clinical Variables

| Variable | Definition |
|----------|-----------|
| Primary Diagnosis | Principal reason for the admission (ICD-9/10 code). Diagnoses like heart failure, COPD, and diabetes carry higher readmission rates. |
| Number of Diagnoses | Total diagnoses coded during the encounter; proxy for clinical complexity. |
| Comorbidity Burden | Number and severity of chronic conditions (e.g., diabetes, renal disease, cancer). Often quantified via the Charlson Comorbidity Index. One of the strongest single predictors. |
| Hemoglobin at Discharge | Blood hemoglobin level (g/dL) at discharge. Low values (anemia) are associated with readmission. |
| Sodium at Discharge | Blood sodium level (mEq/L) at discharge. Low sodium (hyponatremia) is linked to adverse outcomes. |
| A1C Result | Glycated hemoglobin test result; reflects glycemic control in diabetic patients. |
| Max Glucose Serum | Maximum serum glucose recorded during the stay. |
| Procedure During Stay | Whether any procedure was performed during the admission. Indicates severity. |
| Number of Lab Procedures | Count of lab tests performed; higher counts may signal clinical instability. |
| Number of Medications | Total medications prescribed. High medication burden correlates with complexity. |
| Change in Diabetes Medication | Whether diabetic medications were added or changed (yes/no). Signals treatment adjustment. |
| Insulin Prescribed | Whether insulin was part of treatment (yes/no). Indicates diabetes severity. |
| Medical Specialty | Specialty of the admitting physician (e.g., cardiology, oncology). Acts as a proxy for case mix. |

## Utilization Variables

| Variable | Definition |
|----------|-----------|
| Length of Stay | Days in hospital for the index admission. Both very short and very long stays can signal risk. |
| Admission Type | Emergency/urgent vs. elective. Emergency admissions carry higher readmission risk. |
| Prior Inpatient Visits | Hospital admissions in the past 12 months. The strongest utilization-based predictor. |
| Prior ED Visits | Emergency department visits in the prior 6 months. Captures instability and healthcare-seeking behavior. |
| Prior Outpatient Visits | Outpatient encounters in the past 12 months. Low follow-up may indicate poor continuity of care. |
| Discharge Disposition | Where the patient went after discharge (home, skilled nursing facility, against medical advice, etc.). |
| Admission Source | Where the patient was referred from (ED, transfer, physician referral, etc.). |

## Target Variable

| Variable | Definition |
|----------|-----------|
| Readmitted within 30 days | Binary (yes/no). Whether the patient had an unplanned readmission within 30 days of discharge. |
