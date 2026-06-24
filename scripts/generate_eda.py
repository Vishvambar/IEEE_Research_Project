import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("../data/mimic_multidisease_ehr.csv")

# 1. Class Distribution
targets = ['Diabetes_Label', 'CHF_Label', 'Sepsis_Label', 'AKI_Label']
counts = {t: df[t].sum() for t in targets}
total = len(df)
percentages = {t: (counts[t] / total) * 100 for t in targets}

plt.figure(figsize=(10, 6))
bars = plt.bar(percentages.keys(), percentages.values(), color=['skyblue', 'lightgreen', 'salmon', 'orchid'])
plt.title('Disease Prevalence in MIMIC-IV Cohort (N=52,378)', fontsize=14)
plt.ylabel('Prevalence (%)', fontsize=12)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.1f}%', va='bottom', ha='center')
plt.tight_layout()
plt.savefig('../results/EDA_class_distribution.png', dpi=300)
plt.close()

# 2. Correlation Heatmap for continuous features
continuous_features = ['Age', 'heart_rate_avg', 'sys_bp_avg', 'resp_rate_avg', 'spo2_min', 'wbc_max', 'creatinine_max']
plt.figure(figsize=(10, 8))
sns.heatmap(df[continuous_features].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap', fontsize=14)
plt.tight_layout()
plt.savefig('../results/EDA_correlation_heatmap.png', dpi=300)
plt.close()

print("EDA plots generated successfully.")
