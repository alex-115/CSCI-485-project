import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Style ──────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
PALETTE   = "Blues_d"
ACCENT    = "#2c7bb6"
FIG_DIR   = ""
REPORT    = []

def section(title):
    REPORT.append("\n" + "=" * 65)
    REPORT.append(f"  {title}")
    REPORT.append("=" * 65)

def log(msg=""):
    REPORT.append(str(msg))

def savefig(name):
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}{name}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {name}")

# ── Load Data ──────────────────────────────────────────────────────────────
data = pd.read_csv("Hospital_dataset.csv")
data['admission_date'] = pd.to_datetime(data['admission_date'])

NUM_COLS = ['age', 'comorbidities_count', 'length_of_stay',
            'medications_count', 'followup_visits_last_year', 'prev_readmissions']
CAT_COLS = ['season', 'gender', 'region', 'primary_diagnosis',
            'treatment_type', 'insurance_type', 'discharge_disposition']

# ══════════════════════════════════════════════════════════════════════════
# 1. SUMMARY STATISTICS
# ══════════════════════════════════════════════════════════════════════════
section("1. SUMMARY STATISTICS — NUMERIC COLUMNS")
desc = data[NUM_COLS].describe().round(2)
log(desc.to_string())

section("1b. SKEWNESS & KURTOSIS")
for col in NUM_COLS:
    sk = data[col].skew()
    ku = data[col].kurtosis()
    log(f"  {col:<35}  skew={sk:+.3f}   kurtosis={ku:+.3f}")

# ══════════════════════════════════════════════════════════════════════════
# 2. DISTRIBUTIONS — NUMERIC
# ══════════════════════════════════════════════════════════════════════════
print("\n[1/7] Numeric distributions...")
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
for i, col in enumerate(NUM_COLS):
    ax = axes[i]
    sns.histplot(data[col], bins=30, kde=True, ax=ax, color=ACCENT)
    ax.set_title(col.replace("_", " ").title(), fontsize=12, fontweight="bold")
    ax.set_xlabel("")
    mean, median = data[col].mean(), data[col].median()
    ax.axvline(mean,   color="crimson",    linestyle="--", linewidth=1.4, label=f"Mean {mean:.1f}")
    ax.axvline(median, color="darkorange", linestyle=":",  linewidth=1.4, label=f"Median {median:.1f}")
    ax.legend(fontsize=8)
fig.suptitle("Distribution of Numeric Variables", fontsize=15, fontweight="bold", y=1.01)
savefig("01_numeric_distributions.png")

# Box plots
print("[2/7] Box plots...")
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
for i, col in enumerate(NUM_COLS):
    ax = axes[i]
    sns.boxplot(y=data[col], ax=ax, color=ACCENT, width=0.4, flierprops=dict(marker='o', markersize=3, alpha=0.4))
    ax.set_title(col.replace("_", " ").title(), fontsize=12, fontweight="bold")
    ax.set_ylabel("")
fig.suptitle("Box Plots — Numeric Variables (Outlier Detection)", fontsize=15, fontweight="bold", y=1.01)
savefig("02_boxplots.png")

# ══════════════════════════════════════════════════════════════════════════
# 3. CATEGORICAL BREAKDOWNS
# ══════════════════════════════════════════════════════════════════════════
print("[3/7] Categorical breakdowns...")
fig = plt.figure(figsize=(18, 20))
gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35)

plot_positions = [(0,0),(0,1),(1,0),(1,1),(2,0),(2,1),(3,0)]
for pos, col in zip(plot_positions, CAT_COLS):
    ax  = fig.add_subplot(gs[pos])
    vc  = data[col].value_counts()
    colors = sns.color_palette("Blues_d", len(vc))
    bars = ax.barh(vc.index.astype(str), vc.values, color=colors)
    for bar, val in zip(bars, vc.values):
        pct = val / len(data) * 100
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                f"{val:,} ({pct:.1f}%)", va='center', fontsize=8)
    ax.set_title(col.replace("_", " ").title(), fontsize=11, fontweight="bold")
    ax.set_xlabel("Count")
    ax.set_xlim(0, vc.max() * 1.25)

fig.suptitle("Categorical Variable Distributions", fontsize=15, fontweight="bold")
savefig("03_categorical_distributions.png")

section("2. CATEGORICAL VALUE COUNTS")
for col in CAT_COLS:
    vc = data[col].value_counts()
    log(f"\n{col}:")
    for val, cnt in vc.items():
        log(f"  {str(val):<30} {cnt:>6}  ({cnt/len(data)*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════
# 4. CORRELATION MATRIX
# ══════════════════════════════════════════════════════════════════════════
print("[4/7] Correlation matrix...")
corr = data[NUM_COLS].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            linewidths=0.5, ax=ax, square=True,
            annot_kws={"size": 10})
ax.set_title("Correlation Matrix — Numeric Variables", fontsize=14, fontweight="bold", pad=15)
savefig("04_correlation_matrix.png")

section("3. PEARSON CORRELATIONS (pairs |r| > 0.2)")
corr_pairs = (corr.where(np.tril(np.ones(corr.shape), k=-1).astype(bool))
                  .stack()
                  .reset_index())
corr_pairs.columns = ['Var1', 'Var2', 'r']
corr_pairs = corr_pairs.reindex(corr_pairs['r'].abs().sort_values(ascending=False).index)
for _, row in corr_pairs[corr_pairs['r'].abs() > 0.2].iterrows():
    log(f"  {row['Var1']:<30} ↔  {row['Var2']:<30}  r = {row['r']:+.3f}")

# ══════════════════════════════════════════════════════════════════════════
# 5. LENGTH OF STAY — KEY TARGET ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
print("[5/7] Length of stay analysis...")
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# By diagnosis
los_diag = data.groupby('primary_diagnosis')['length_of_stay'].mean().sort_values(ascending=False)
axes[0,0].barh(los_diag.index, los_diag.values, color=sns.color_palette("Blues_d", len(los_diag)))
axes[0,0].set_title("Avg Length of Stay by Diagnosis", fontweight="bold")
axes[0,0].set_xlabel("Days")
for i, v in enumerate(los_diag.values):
    axes[0,0].text(v + 0.05, i, f"{v:.1f}", va='center', fontsize=8)

# By treatment type
los_treat = data.groupby('treatment_type')['length_of_stay'].mean().sort_values(ascending=False)
axes[0,1].bar(los_treat.index, los_treat.values, color=sns.color_palette("Blues_d", len(los_treat)))
axes[0,1].set_title("Avg Length of Stay by Treatment", fontweight="bold")
axes[0,1].set_ylabel("Days")
for i, (x, v) in enumerate(zip(los_treat.index, los_treat.values)):
    axes[0,1].text(i, v + 0.05, f"{v:.1f}", ha='center', fontsize=9)

# By insurance type
los_ins = data.groupby('insurance_type')['length_of_stay'].mean().sort_values(ascending=False)
axes[0,2].bar(los_ins.index, los_ins.values, color=sns.color_palette("Blues_d", len(los_ins)))
axes[0,2].set_title("Avg Length of Stay by Insurance", fontweight="bold")
axes[0,2].set_ylabel("Days")
for i, (x, v) in enumerate(zip(los_ins.index, los_ins.values)):
    axes[0,2].text(i, v + 0.05, f"{v:.1f}", ha='center', fontsize=9)

# LOS vs age scatter
axes[1,0].scatter(data['age'], data['length_of_stay'], alpha=0.15, s=8, color=ACCENT)
m, b, r, p, _ = stats.linregress(data['age'], data['length_of_stay'])
x_line = np.linspace(data['age'].min(), data['age'].max(), 200)
axes[1,0].plot(x_line, m*x_line + b, color='crimson', linewidth=2, label=f"r={r:.2f}, p={p:.3f}")
axes[1,0].set_title("Length of Stay vs Age", fontweight="bold")
axes[1,0].set_xlabel("Age"); axes[1,0].set_ylabel("Days")
axes[1,0].legend(fontsize=9)

# LOS vs comorbidities box
data.boxplot(column='length_of_stay', by='comorbidities_count', ax=axes[1,1],
             boxprops=dict(color=ACCENT), medianprops=dict(color='crimson', linewidth=2))
axes[1,1].set_title("LOS by Comorbidity Count", fontweight="bold")
axes[1,1].set_xlabel("Comorbidities"); axes[1,1].set_ylabel("Days")
plt.sca(axes[1,1]); plt.title("LOS by Comorbidity Count"); plt.suptitle("")

# LOS by season
los_season = data.groupby('season')['length_of_stay'].mean()
order = ['Spring','Summer','Fall','Winter']
vals  = [los_season.get(s, 0) for s in order]
axes[1,2].bar(order, vals, color=sns.color_palette("Blues_d", 4))
axes[1,2].set_title("Avg Length of Stay by Season", fontweight="bold")
axes[1,2].set_ylabel("Days")
for i, v in enumerate(vals):
    axes[1,2].text(i, v + 0.05, f"{v:.1f}", ha='center', fontsize=9)

fig.suptitle("Length of Stay — Deep Dive", fontsize=15, fontweight="bold")
plt.tight_layout()
savefig("05_length_of_stay_analysis.png")

section("4. LENGTH OF STAY — GROUP MEANS")
for col in ['primary_diagnosis', 'treatment_type', 'insurance_type', 'region', 'season']:
    log(f"\nBy {col}:")
    gm = data.groupby(col)['length_of_stay'].agg(['mean','median','std']).round(2)
    gm.columns = ['Mean', 'Median', 'Std']
    gm = gm.sort_values('Mean', ascending=False)
    log(gm.to_string())

# ══════════════════════════════════════════════════════════════════════════
# 6. READMISSION & RISK ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
print("[6/7] Readmission analysis...")
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# Readmission distribution
vc_ra = data['prev_readmissions'].value_counts().sort_index()
axes[0,0].bar(vc_ra.index.astype(str), vc_ra.values, color=sns.color_palette("Blues_d", len(vc_ra)))
axes[0,0].set_title("Distribution of Previous Readmissions", fontweight="bold")
axes[0,0].set_xlabel("Count"); axes[0,0].set_ylabel("Patients")
for i, (x, v) in enumerate(zip(vc_ra.index, vc_ra.values)):
    axes[0,0].text(i, v + 20, f"{v}", ha='center', fontsize=9)

# Avg readmissions by diagnosis
ra_diag = data.groupby('primary_diagnosis')['prev_readmissions'].mean().sort_values(ascending=False)
axes[0,1].barh(ra_diag.index, ra_diag.values, color=sns.color_palette("Blues_d", len(ra_diag)))
axes[0,1].set_title("Avg Readmissions by Diagnosis", fontweight="bold")
axes[0,1].set_xlabel("Avg Previous Readmissions")
for i, v in enumerate(ra_diag.values):
    axes[0,1].text(v + 0.01, i, f"{v:.2f}", va='center', fontsize=8)

# Readmissions vs age
ra_age = data.groupby(pd.cut(data['age'], bins=[0,34,49,64,79,120],
                              labels=['≤34','35-49','50-64','65-79','80+']))['prev_readmissions'].mean()
axes[0,2].bar(ra_age.index.astype(str), ra_age.values, color=sns.color_palette("Blues_d", len(ra_age)))
axes[0,2].set_title("Avg Readmissions by Age Group", fontweight="bold")
axes[0,2].set_xlabel("Age Group"); axes[0,2].set_ylabel("Avg Readmissions")
for i, v in enumerate(ra_age.values):
    axes[0,2].text(i, v + 0.01, f"{v:.2f}", ha='center', fontsize=9)

# Readmissions by insurance
ra_ins = data.groupby('insurance_type')['prev_readmissions'].mean().sort_values(ascending=False)
axes[1,0].bar(ra_ins.index, ra_ins.values, color=sns.color_palette("Blues_d", len(ra_ins)))
axes[1,0].set_title("Avg Readmissions by Insurance", fontweight="bold")
axes[1,0].set_ylabel("Avg Previous Readmissions")
for i, v in enumerate(ra_ins.values):
    axes[1,0].text(i, v + 0.01, f"{v:.2f}", ha='center', fontsize=9)

# Comorbidities vs medications scatter
axes[1,1].scatter(data['comorbidities_count'], data['medications_count'],
                  alpha=0.15, s=8, color=ACCENT)
m, b, r, p, _ = stats.linregress(data['comorbidities_count'], data['medications_count'])
x_line = np.linspace(data['comorbidities_count'].min(), data['comorbidities_count'].max(), 100)
axes[1,1].plot(x_line, m*x_line + b, color='crimson', linewidth=2, label=f"r={r:.2f}")
axes[1,1].set_title("Comorbidities vs Medications Count", fontweight="bold")
axes[1,1].set_xlabel("Comorbidities"); axes[1,1].set_ylabel("Medications")
axes[1,1].legend(fontsize=9)

# Discharge disposition by insurance (stacked %)
pivot = pd.crosstab(data['insurance_type'], data['discharge_disposition'], normalize='index') * 100
pivot.plot(kind='bar', stacked=True, ax=axes[1,2],
           colormap='Blues', edgecolor='white', linewidth=0.5)
axes[1,2].set_title("Discharge Disposition by Insurance (%)", fontweight="bold")
axes[1,2].set_xlabel(""); axes[1,2].set_ylabel("% Patients")
axes[1,2].legend(fontsize=7, loc='upper right')
axes[1,2].tick_params(axis='x', rotation=30)

fig.suptitle("Readmission & Clinical Risk Analysis", fontsize=15, fontweight="bold")
plt.tight_layout()
savefig("06_readmission_analysis.png")

# ══════════════════════════════════════════════════════════════════════════
# 7. TEMPORAL TRENDS
# ══════════════════════════════════════════════════════════════════════════
print("[7/7] Temporal trends...")
data['yr_month'] = data['admission_date'].dt.to_period('M')
monthly = data.groupby('yr_month').agg(
    admissions=('patient_id','count'),
    avg_los=('length_of_stay','mean'),
    avg_readmissions=('prev_readmissions','mean')
).reset_index()
monthly['yr_month_dt'] = monthly['yr_month'].dt.to_timestamp()

fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
for ax, col, label, color in zip(
    axes,
    ['admissions','avg_los','avg_readmissions'],
    ['Monthly Admissions','Avg Length of Stay (days)','Avg Prior Readmissions'],
    [ACCENT, 'darkorange', 'crimson']
):
    ax.plot(monthly['yr_month_dt'], monthly[col], color=color, linewidth=2, marker='o', markersize=4)
    ax.fill_between(monthly['yr_month_dt'], monthly[col], alpha=0.15, color=color)
    ax.set_ylabel(label, fontsize=10)
    ax.grid(axis='y', alpha=0.4)

axes[0].set_title("Temporal Trends (Monthly)", fontsize=14, fontweight="bold")
axes[2].set_xlabel("Month")
savefig("07_temporal_trends.png")

section("5. TEMPORAL SUMMARY")
log(f"Date range: {data['admission_date'].min().date()} → {data['admission_date'].max().date()}")
log(f"Years covered: {sorted(data['admission_date'].dt.year.unique())}")
yr_counts = data.groupby(data['admission_date'].dt.year)['patient_id'].count()
log("\nAdmissions per year:")
for yr, cnt in yr_counts.items():
    log(f"  {yr}: {cnt}")

# ══════════════════════════════════════════════════════════════════════════
# SAVE REPORT
# ══════════════════════════════════════════════════════════════════════════
section("6. KEY TAKEAWAYS")
log(f"  • Dataset: {len(data):,} patients, {data.shape[1]} variables")
log(f"  • Avg age: {data['age'].mean():.1f} yrs  |  Range: {data['age'].min()}–{data['age'].max()}")
log(f"  • Avg length of stay: {data['length_of_stay'].mean():.1f} days")
log(f"  • Most common diagnosis: {data['primary_diagnosis'].value_counts().index[0]}")
log(f"  • Most common treatment: {data['treatment_type'].value_counts().index[0]}")
log(f"  • Patients with 0 prior readmissions: {(data['prev_readmissions']==0).sum():,} ({(data['prev_readmissions']==0).mean()*100:.1f}%)")
log(f"  • Patients with 3+ prior readmissions: {(data['prev_readmissions']>=3).sum():,} ({(data['prev_readmissions']>=3).mean()*100:.1f}%)")
corr_los_comor = data['length_of_stay'].corr(data['comorbidities_count'])
corr_los_age   = data['length_of_stay'].corr(data['age'])
log(f"  • Correlation: LOS ↔ comorbidities = {corr_los_comor:.3f}")
log(f"  • Correlation: LOS ↔ age           = {corr_los_age:.3f}")

report_text = "\n".join(REPORT)
with open("EDA_Report.txt", "w", encoding="utf-8") as f:
    f.write("HOSPITAL DATASET — EXPLORATORY DATA ANALYSIS REPORT\n")
    f.write(f"Dataset: Hospital_dataset.csv  |  N = {len(data):,} patients\n")
    f.write(report_text)

print("\n✓ EDA complete. Files saved:")
print("  EDA_Report.txt")
print("  01_numeric_distributions.png")
print("  02_boxplots.png")
print("  03_categorical_distributions.png")
print("  04_correlation_matrix.png")
print("  05_length_of_stay_analysis.png")
print("  06_readmission_analysis.png")
print("  07_temporal_trends.png")