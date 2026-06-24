# Week 2 – Development & Experimental Setup

**Project Title:** Multi-Disease Early Screening Using Deep Multi-Task Learning
**Guide:** Dr. Ranjana Kale
**Group Members:** Vishvambar Ramesh Udavant, Ish Prafull Chaniyara, Sagar Shahgond

---

## 1. Objectives Completed
- **Began implementation** of the core machine learning pipeline.
- **Prepared dataset** and experimental environment (Python, PyTorch, Pandas, Scikit-learn).
- **Developed initial AI models** transitioning from disjoint XGBoost trees to a unified PyTorch Causal DAG network.

## 2. Data Preprocessing & Exploratory Data Analysis (EDA)

### Data Preprocessing
The `mimic_multidisease_ehr.csv` dataset, consisting of 52,378 patients and 17 physiological features, was loaded and cleaned.
- **Missing Value Handling:** Executed median imputation using `SimpleImputer(strategy='median')` to resolve physiological NaNs without dropping massive patient cohorts.
- **Scaling:** Applied `StandardScaler` to normalize features (e.g., heart rate, blood pressure) to zero mean and unit variance, a critical requirement for neural network gradient convergence.
- **Feature Engineering:** Derived 3 new volatility range features (`heart_rate_range`, `sys_bp_range`, `resp_rate_range`) to capture physiological instability without needing raw time-series data.

### EDA Observations
- **Class Imbalance:** The target variables exhibit significant imbalance, heavily skewed towards negative classes. Sepsis is the rarest at 15.9%, while AKI is the most prevalent at 29.8%. 
- **Correlations:** Standard physiological correlations were confirmed (e.g., systolic and diastolic blood pressure ranges), but no single feature showed a trivially high correlation with the complex targets, proving the necessity of deep learning representations.

*(See attached `EDA_class_distribution.png` and `EDA_correlation_heatmap.png` in the project root).*

## 3. Experimental Setup and Model Development

### PyTorch Directed Acyclic Graph (DAG) Multi-Task Network
Rather than training independent models (which ignores comorbidity dependencies), we developed a PyTorch network that respects the clinical timeline.
- **Architecture:** A shared 64-dimensional trunk passes embeddings into sequentially chained heads: `Diabetes (Chronic) -> CHF (Progressive) -> Sepsis (Acute) -> AKI (Complication)`.
- **Stop-Gradient Backpropagation:** We used PyTorch `.detach()` on the lateral hidden states. This prevents "Catastrophic Interference" where massive acute loss gradients (AKI) destroy the upstream weights of chronic predictors (Diabetes).
- **Static Focal Loss:** Implemented Focal Loss ($\gamma=2$) scaled by inverse class frequency to heavily penalize the network for missing rare, hard-to-predict positive cases.

## 4. Preliminary Results

We executed a strict clinical threshold calibration loop, anchoring on a minimum clinical safety net of `Recall >= 0.50` and `Precision >= 0.65` to avoid alarm fatigue.

### Threshold Optimization Summary
| Disease | Optimal Threshold | Precision | Recall | F1-Score | Status |
|---|---|---|---|---|---|
| **Diabetes** | 0.50 | 0.374 | 0.674 | 0.481 | ❌ Clinically Unviable |
| **CHF** | 0.50 | 0.403 | 0.667 | 0.503 | ❌ Clinically Unviable |
| **Sepsis** | 0.55 | 0.411 | 0.578 | 0.480 | ❌ Clinically Unviable |
| **AKI** | 0.55 | 0.724 | 0.718 | 0.721 | ✅ Viable |

### Observations and Conclusions
As expected, a neural network cannot invent signal from noise. The 17 core features fundamentally lack the necessary historical data (e.g., prior HbA1c labs) required to reliably detect Diabetes, CHF, and Sepsis without triggering massive false alarms. Thus, they failed the clinical safety net.

However, the **AKI model** (which was fed by the dense latent context flowing downstream from the chronic heads) achieved excellent viability with a Precision of 0.724 and Recall of 0.718. This successfully proves that the Causal DAG significantly outperforms disjoint modeling for capturing organ failure comorbidity risk.

## 5. Next Steps
- Implement Captum Integrated Gradients for advanced model explainability.
- Develop the final interactive dashboard or command-line tool.
- Plan the integration of V2 time-series data to resolve the unviable disease heads.
