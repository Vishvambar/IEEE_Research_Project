import torch
import numpy as np
import pandas as pd
from mtl_model import DAG_MTL_Model
from captum.attr import IntegratedGradients

df = pd.read_csv("../data/mimic_multidisease_ehr.csv")
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

X = df[feature_cols]
Y = df[['Diabetes_Label', 'CHF_Label', 'Sepsis_Label', 'AKI_Label']]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
imputer = SimpleImputer(strategy='median')
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_cols)
X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=feature_cols)
scaler = StandardScaler()
X_train_scl = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=feature_cols)
X_test_scl = pd.DataFrame(scaler.transform(X_test_imp), columns=feature_cols)

device = torch.device("cpu")
model = DAG_MTL_Model().to(device)
# Note: we need the trained model weights. But we didn't save the weights in train_mtl.py!
# Oh no, train_mtl.py did not save the model weights.
