import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from mtl_model import DAG_MTL_Model
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# 1. Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha 
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * bce_loss
        
        if self.alpha is not None:
            alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            focal_loss = alpha_t * focal_loss
            
        return focal_loss.mean(dim=0).sum()

# 2. Dataset
class MIMICDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 3. Data Prep
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
target_cols = ['Diabetes_Label', 'CHF_Label', 'Sepsis_Label', 'AKI_Label']

X = df[feature_cols]
Y = df[target_cols]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

imputer = SimpleImputer(strategy='median')
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_cols)
X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=feature_cols)

scaler = StandardScaler()
X_train_scl = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=feature_cols)
X_test_scl = pd.DataFrame(scaler.transform(X_test_imp), columns=feature_cols)

alphas = 1.0 - (Y_train.mean().values) 
alpha_tensor = torch.tensor(alphas, dtype=torch.float32)

train_dataset = MIMICDataset(X_train_scl, Y_train)
test_dataset = MIMICDataset(X_test_scl, Y_test)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# 4. Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DAG_MTL_Model().to(device)
criterion = FocalLoss(alpha=alpha_tensor.to(device), gamma=2.0)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

epochs = 15
print(f"Training on {device}...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")

# 5. Evaluation
model.eval()
all_preds = []
all_targets = []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        logits = model(batch_x)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(probs)
        all_targets.append(batch_y.numpy())

y_prob = np.vstack(all_preds)
y_true = np.vstack(all_targets)

# 6. Threshold Optimization
print("\n========== THRESHOLD CALIBRATION ==========")
thresholds = np.arange(0.05, 1.00, 0.05)
for i, disease in enumerate(target_cols):
    y_t = y_true[:, i]
    y_p = y_prob[:, i]
    
    roc_auc = roc_auc_score(y_t, y_p)
    pr_auc = average_precision_score(y_t, y_p)
    
    valid_thresholds = []
    for t in thresholds:
        y_pred = (y_p >= t).astype(int)
        p = precision_score(y_t, y_pred, zero_division=0)
        r = recall_score(y_t, y_pred, zero_division=0)
        f1 = f1_score(y_t, y_pred, zero_division=0)
        valid_thresholds.append({'t': t, 'p': p, 'r': r, 'f1': f1})
        
    both_constrained = [x for x in valid_thresholds if x['p'] >= 0.65 and x['r'] >= 0.50]
    if len(both_constrained) > 0:
        best = max(both_constrained, key=lambda x: x['f1'])
        note = "Viable (P>=0.65, R>=0.50)"
    else:
        r_constrained = [x for x in valid_thresholds if x['r'] >= 0.50]
        if len(r_constrained) > 0:
            best = max(r_constrained, key=lambda x: x['p'])
            note = "Unviable (Max P < 0.65 at R>=0.5)"
        else:
            best = max(valid_thresholds, key=lambda x: x['f1'])
            note = "Completely Unviable"
            
    print(f"{disease:15} | ROC-AUC: {roc_auc:.3f} | PR-AUC: {pr_auc:.3f} | {note:25} | Threshold: {best['t']:.2f} | Precision: {best['p']:.3f} | Recall: {best['r']:.3f} | F1: {best['f1']:.3f}")

# 7. Captum Explainability for AKI
print("\nGenerating Captum IntegratedGradients for AKI...")

class AKIWrapper(nn.Module):
    def __init__(self, mtl_model):
        super().__init__()
        self.mtl_model = mtl_model
    def forward(self, x):
        return self.mtl_model(x)[:, 3].unsqueeze(1) 

aki_wrapper = AKIWrapper(model).to(device)
ig = IntegratedGradients(aki_wrapper)

sample = torch.tensor(X_test_scl.iloc[[0]].values, dtype=torch.float32).to(device)
baseline = torch.zeros_like(sample).to(device)

attributions, delta = ig.attribute(sample, baseline, return_convergence_delta=True)
attributions = attributions.cpu().numpy()[0]

plt.figure(figsize=(10, 6))
y_pos = np.arange(len(feature_cols))
sorted_idx = np.argsort(np.abs(attributions))
plt.barh(y_pos, attributions[sorted_idx], align='center')
plt.yticks(y_pos, np.array(feature_cols)[sorted_idx])
plt.xlabel('Integrated Gradients Attribution')
plt.title('Captum Feature Attributions for AKI Prediction (Patient 0)')
plt.tight_layout()
plt.savefig("../results/Captum_Attributions_AKI.png", dpi=300)
plt.close()
print("Saved ../results/Captum_Attributions_AKI.png")

print("Top 5 Features:")
top_indices = np.argsort(np.abs(attributions))[::-1][:5]
for i, idx in enumerate(top_indices):
    print(f"Rank {i+1}: {feature_cols[idx]} ({attributions[idx]:.4f})")
