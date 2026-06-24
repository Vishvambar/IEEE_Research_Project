import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.multioutput import ClassifierChain
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import shap

# Load Data
df = pd.read_csv("../data/mimic_multidisease_ehr.csv")

# Feature Engineering
df['heart_rate_range'] = df['heart_rate_max'] - df['heart_rate_min']
df['sys_bp_range'] = df['sys_bp_max'] - df['sys_bp_min']
df['resp_rate_range'] = df['resp_rate_max'] - df['resp_rate_min']

feature_cols = [
    'Age', 'Gender',
    'heart_rate_min', 'heart_rate_max', 'heart_rate_avg', 'heart_rate_range',
    'sys_bp_min', 'sys_bp_max', 'sys_bp_avg', 'sys_bp_range',
    'resp_rate_min', 'resp_rate_max', 'resp_rate_avg', 'resp_rate_range',
    'spo2_min',
    'wbc_max', 'creatinine_max'
]

# Correct Clinical Order
target_cols = [
    'Diabetes_Label',
    'CHF_Label',
    'Sepsis_Label',
    'AKI_Label'
]

X = df[feature_cols]
Y = df[target_cols]

# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.20, random_state=42
)

# Impute NaNs because ClassifierChain does not accept them natively
imputer = SimpleImputer(strategy='median')
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_cols, index=X_train.index)
X_test = pd.DataFrame(imputer.transform(X_test), columns=feature_cols, index=X_test.index)

print(f"Training Patients : {len(X_train):,}")
print(f"Testing Patients  : {len(X_test):,}")

# Model
base_model = XGBClassifier(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=5,
    tree_method='hist',
    eval_metric='logloss',
    random_state=42
)

# ClassifierChain respects the order of targets in Y automatically if order=None
model = ClassifierChain(base_model, order=None)

print("\\nTraining model...")
model.fit(X_train, Y_train)

# Probabilities
prob_matrix = model.predict_proba(X_test)

# Threshold Optimization
thresholds = np.arange(0.05, 1.00, 0.05)
results = []
best_thresholds_dict = {}

print("\\n========== THRESHOLD CALIBRATION ==========\\n")

for i, disease in enumerate(target_cols):
    y_true = Y_test.iloc[:, i]
    y_prob = prob_matrix[:, i]
    
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    
    best_threshold = 0.5
    best_precision = 0
    best_recall = 0
    best_f1 = 0
    
    # Logic: maximize Recall subject to Precision >= 0.65
    # Fallback: if impossible, maximize Precision subject to Recall >= 0.50
    valid_thresholds = []
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        valid_thresholds.append({'t': t, 'p': p, 'r': r, 'f1': f1})
    
    # Try to find a threshold that satisfies BOTH P >= 0.65 AND R >= 0.50
    both_constrained = [x for x in valid_thresholds if x['p'] >= 0.65 and x['r'] >= 0.50]
    
    if len(both_constrained) > 0:
        best = max(both_constrained, key=lambda x: x['f1'])
        note = "Viable (P>=0.65, R>=0.50)"
    else:
        # Fallback: R >= 0.50
        r_constrained = [x for x in valid_thresholds if x['r'] >= 0.50]
        if len(r_constrained) > 0:
            best = max(r_constrained, key=lambda x: x['p'])
            note = "Unviable (Max P < 0.65 at R>=0.5)"
        else:
            best = max(valid_thresholds, key=lambda x: x['f1'])
            note = "Completely Unviable"
            
    best_threshold = best['t']
    best_precision = best['p']
    best_recall = best['r']
    best_f1 = best['f1']
    best_thresholds_dict[disease] = best_threshold
    
    results.append([
        disease, roc_auc, pr_auc, best_threshold, best_precision, best_recall, best_f1, note
    ])
    
    # Plotting
    plt.figure(figsize=(7,5))
    f1_curve = [x['f1'] for x in valid_thresholds]
    plt.plot(thresholds, f1_curve, label='F1 Score')
    plt.axvline(best_threshold, linestyle='--', color='red', label=f'Chosen ({best_threshold:.2f})')
    plt.title(f"{disease} Threshold Optimization")
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"../results/{disease}_threshold_curve.png", dpi=300)
    plt.close()

results_df = pd.DataFrame(results, columns=["Disease", "ROC_AUC", "PR_AUC", "Optimal_Threshold", "Precision", "Recall", "F1", "Status"])
print(results_df.round(3))
results_df.to_csv("../results/threshold_calibration_results.csv", index=False)

print("\\n========== FINAL CLINICAL SCREENING RESULTS ==========\\n")

for i, disease in enumerate(target_cols):
    y_true = Y_test.iloc[:, i]
    y_prob = prob_matrix[:, i]
    threshold = best_thresholds_dict[disease]
    y_pred = (y_prob >= threshold).astype(int)
    
    print(f"\\n===== {disease} =====")
    print(f"Threshold Used: {threshold:.2f}")
    print(classification_report(y_true, y_pred, digits=3))
    print("Confusion Matrix")
    print(confusion_matrix(y_true, y_pred))

# ==========================================================
# SHAP EXPLAINABILITY
# ==========================================================
print("\\nGenerating SHAP Explanations...")

aki_features = feature_cols + ['Diabetes_Label', 'CHF_Label', 'Sepsis_Label']
X_test_aki_chain = np.zeros((X_test.shape[0], len(aki_features)))
X_test_aki_chain[:, :len(feature_cols)] = X_test.values

previous_preds = np.zeros((X_test.shape[0], 3))
X_temp = X_test.values
for i in range(3):
    pred = model.estimators_[i].predict(X_temp)
    previous_preds[:, i] = pred
    X_temp = np.column_stack([X_temp, pred])

X_test_aki_chain[:, len(feature_cols):] = previous_preds
df_aki_test = pd.DataFrame(X_test_aki_chain, columns=aki_features)

aki_model = model.estimators_[3]
explainer = shap.TreeExplainer(aki_model)
shap_values = explainer(df_aki_test)

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, df_aki_test, show=False)
plt.tight_layout()
plt.savefig("../results/SHAP_summary_AKI.png", bbox_inches='tight', dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap_values[0], show=False)
plt.tight_layout()
plt.savefig("../results/SHAP_waterfall_AKI_patient_0.png", bbox_inches='tight', dpi=300)
plt.close()

print("SHAP plots saved in ../results/")
