# Multi-Disease Early Screening Using XGBoost on MIMIC-IV EHR Data

> **IEEE Research Project** — A multi-label clinical decision support system for simultaneous early detection of Sepsis, Acute Kidney Injury (AKI), Congestive Heart Failure (CHF), and Diabetes from Electronic Health Records.

---

## Table of Contents

- [Overview](#overview)
- [Clinical Motivation](#clinical-motivation)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Model Architecture](#model-architecture)
- [Threshold Calibration](#threshold-calibration)
- [Results](#results)
  - [Threshold Optimization Summary](#threshold-optimization-summary)
  - [Per-Disease Classification Reports](#per-disease-classification-reports)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Model](#running-the-model)
- [Methodology](#methodology)
- [Limitations & Future Work](#limitations--future-work)
- [License](#license)

---

## Overview

This project presents a **multi-output XGBoost classifier** designed to simultaneously predict the risk of four critical diseases from a single patient encounter using structured EHR (Electronic Health Record) data derived from the **MIMIC-IV** clinical database.

Unlike traditional single-disease models, this system performs **multi-label classification** — a single forward pass produces independent risk probabilities for all four conditions, enabling holistic patient screening in resource-constrained clinical environments.

The pipeline includes:

1. **Data ingestion** from a preprocessed MIMIC-IV cohort
2. **Multi-output XGBoost training** with histogram-based tree construction
3. **Per-disease threshold calibration** via F1-score maximization
4. **Clinical evaluation** with classification reports and confusion matrices

---

## Clinical Motivation

In critical care settings, patients often present with **overlapping and co-occurring conditions**. Sepsis may trigger AKI, diabetic patients have elevated CHF risk, and these comorbidities interact in complex ways. Current clinical workflows typically rely on **disease-specific screening tools** applied independently, which:

- Increases clinician cognitive load
- Delays multi-condition diagnosis
- Misses comorbidity interactions

This project addresses these challenges by providing a **unified screening model** that simultaneously flags patients at risk for multiple conditions using routinely collected vital signs and lab values — data already available at the point of care.

---

## Dataset

| Property               | Value                                      |
|------------------------|---------------------------------------------|
| **Source**             | MIMIC-IV Clinical Database                  |
| **File**               | `mimic_multidisease_ehr.csv`               |
| **Total Patients**     | 52,378                                     |
| **Training Set**       | 41,902 patients (80%)                      |
| **Test Set**           | 10,476 patients (20%)                      |
| **Split Strategy**     | Random split with `random_state=42`        |
| **Label Type**         | Multi-label binary (4 independent targets) |

### Target Distribution (Test Set)

| Disease        | Positive Cases | Negative Cases | Prevalence |
|---------------|---------------|---------------|------------|
| Sepsis         | 1,663         | 8,813          | 15.9%      |
| AKI            | 3,117         | 7,359          | 29.8%      |
| CHF            | 2,389         | 8,087          | 22.8%      |
| Diabetes       | 2,827         | 7,649          | 27.0%      |

---

## Feature Engineering

The model uses **14 clinically interpretable features** derived from routinely collected patient data:

### Demographics (2 features)

| Feature  | Description                                  |
|----------|----------------------------------------------|
| `Age`    | Patient age at admission                     |
| `Gender` | Binary-encoded sex (0 = Female, 1 = Male)   |

### Vital Signs (10 features)

| Feature           | Description                                         |
|-------------------|-----------------------------------------------------|
| `heart_rate_min`  | Minimum heart rate during ICU stay (bpm)            |
| `heart_rate_max`  | Maximum heart rate during ICU stay (bpm)            |
| `heart_rate_avg`  | Mean heart rate during ICU stay (bpm)               |
| `sys_bp_min`      | Minimum systolic blood pressure (mmHg)              |
| `sys_bp_max`      | Maximum systolic blood pressure (mmHg)              |
| `sys_bp_avg`      | Mean systolic blood pressure (mmHg)                 |
| `resp_rate_min`   | Minimum respiratory rate (breaths/min)              |
| `resp_rate_max`   | Maximum respiratory rate (breaths/min)              |
| `resp_rate_avg`   | Mean respiratory rate (breaths/min)                 |
| `spo2_min`        | Minimum peripheral oxygen saturation (%)            |

### Laboratory Values (2 features)

| Feature           | Description                                         |
|-------------------|-----------------------------------------------------|
| `wbc_max`         | Peak white blood cell count (×10³/µL)               |
| `creatinine_max`  | Peak serum creatinine level (mg/dL)                 |

> **Design Rationale:** Features were selected for clinical relevance and universal availability in ICU settings. Min/max/avg aggregations capture both acute derangements and sustained abnormalities, which are critical for distinguishing between transient physiological stress and true disease onset.

---

## Model Architecture

```
┌──────────────────────────────────────────────────────┐
│                  Input: 14 Features                  │
│  (Age, Gender, Vitals, Labs)                         │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│          MultiOutputClassifier Wrapper               │
│  (sklearn.multioutput.MultiOutputClassifier)         │
│                                                      │
│  ┌────────────┐ ┌────────────┐ ┌───────┐ ┌────────┐ │
│  │ XGBoost #1 │ │ XGBoost #2 │ │ XGB#3 │ │ XGB #4 │ │
│  │  (Sepsis)  │ │   (AKI)    │ │ (CHF) │ │(Diab.) │ │
│  └─────┬──────┘ └─────┬──────┘ └───┬───┘ └───┬────┘ │
│        │              │            │          │      │
└────────┼──────────────┼────────────┼──────────┼──────┘
         ▼              ▼            ▼          ▼
   ┌──────────┐   ┌──────────┐ ┌────────┐ ┌────────┐
   │ P(Sepsis)│   │  P(AKI)  │ │ P(CHF) │ │P(Diab.)│
   └──────────┘   └──────────┘ └────────┘ └────────┘
         │              │            │          │
         ▼              ▼            ▼          ▼
   ┌──────────────────────────────────────────────────┐
   │       Per-Disease Threshold Calibration          │
   │  Sepsis ≥ 0.25 │ AKI ≥ 0.40 │ CHF ≥ 0.25 │     │
   │                 │            │ Diab. ≥ 0.25│     │
   └──────────────────────────────────────────────────┘
         │              │            │          │
         ▼              ▼            ▼          ▼
   ┌──────────────────────────────────────────────────┐
   │         Binary Predictions per Disease           │
   │        (0 = Low Risk, 1 = High Risk)             │
   └──────────────────────────────────────────────────┘
```

### XGBoost Hyperparameters

| Parameter        | Value     | Rationale                                              |
|-----------------|-----------|--------------------------------------------------------|
| `n_estimators`   | 150       | Sufficient ensemble depth without overfitting          |
| `learning_rate`  | 0.05      | Conservative step size for stable convergence          |
| `max_depth`      | 5         | Captures non-linear feature interactions               |
| `tree_method`    | `hist`    | Histogram-based splits for faster training on large data|
| `eval_metric`    | `logloss` | Probabilistic calibration-friendly loss function       |
| `random_state`   | 42        | Reproducibility                                       |

---

## Threshold Calibration

Clinical screening models require **disease-specific decision thresholds** rather than a default 0.5 cutoff. This is because:

- **Class imbalance** varies across diseases (e.g., Sepsis at 15.9% vs AKI at 29.8%)
- **Clinical cost asymmetry** — missing a sepsis case (false negative) is far more dangerous than a false alarm

The pipeline sweeps thresholds from **0.05 to 0.95** in increments of 0.05, computing F1-score at each point. The threshold maximizing F1 is selected per disease.

For each disease, a **Threshold vs. F1 curve** is generated and saved as a PNG file.

---

## Results

### Threshold Optimization Summary

| Disease        | ROC-AUC | PR-AUC | Optimal Threshold | Precision | Recall | F1-Score |
|---------------|---------|--------|-------------------|-----------|--------|----------|
| **Sepsis**     | 0.820   | 0.484  | 0.25              | 0.440     | 0.583  | 0.501    |
| **AKI**        | 0.885   | 0.770  | 0.40              | 0.718     | 0.738  | 0.728    |
| **CHF**        | 0.758   | 0.489  | 0.25              | 0.405     | 0.679  | 0.507    |
| **Diabetes**   | 0.683   | 0.434  | 0.25              | 0.362     | 0.734  | 0.485    |

### Per-Disease Classification Reports

#### Sepsis (Threshold = 0.25)

| Metric         | Class 0 (No Sepsis) | Class 1 (Sepsis) |
|---------------|--------------------:|------------------:|
| Precision      | 0.916              | 0.440             |
| Recall         | 0.860              | 0.583             |
| F1-Score       | 0.887              | 0.501             |
| **Support**    | 8,813              | 1,663             |

| **Overall Accuracy** | **0.816** |
|---------------------|-----------|

**Confusion Matrix:**

|                   | Predicted Negative | Predicted Positive |
|-------------------|-------------------:|-------------------:|
| Actual Negative   | 7,578              | 1,235              |
| Actual Positive   | 694                | 969                |

---

#### AKI — Acute Kidney Injury (Threshold = 0.40)

| Metric         | Class 0 (No AKI) | Class 1 (AKI) |
|---------------|------------------:|---------------:|
| Precision      | 0.888            | 0.718          |
| Recall         | 0.877            | 0.738          |
| F1-Score       | 0.883            | 0.728          |
| **Support**    | 7,359            | 3,117          |

| **Overall Accuracy** | **0.836** |
|---------------------|-----------|

**Confusion Matrix:**

|                   | Predicted Negative | Predicted Positive |
|-------------------|-------------------:|-------------------:|
| Actual Negative   | 6,456              | 903                |
| Actual Positive   | 816                | 2,301              |

---

#### CHF — Congestive Heart Failure (Threshold = 0.25)

| Metric         | Class 0 (No CHF) | Class 1 (CHF) |
|---------------|------------------:|---------------:|
| Precision      | 0.881            | 0.405          |
| Recall         | 0.705            | 0.679          |
| F1-Score       | 0.783            | 0.507          |
| **Support**    | 8,087            | 2,389          |

| **Overall Accuracy** | **0.699** |
|---------------------|-----------|

**Confusion Matrix:**

|                   | Predicted Negative | Predicted Positive |
|-------------------|-------------------:|-------------------:|
| Actual Negative   | 5,701              | 2,386              |
| Actual Positive   | 767                | 1,622              |

---

#### Diabetes (Threshold = 0.25)

| Metric         | Class 0 (No Diabetes) | Class 1 (Diabetes) |
|---------------|-----------------------:|-------------------:|
| Precision      | 0.841                 | 0.362              |
| Recall         | 0.522                 | 0.734              |
| F1-Score       | 0.644                 | 0.485              |
| **Support**    | 7,649                 | 2,827              |

| **Overall Accuracy** | **0.579** |
|---------------------|-----------|

**Confusion Matrix:**

|                   | Predicted Negative | Predicted Positive |
|-------------------|-------------------:|-------------------:|
| Actual Negative   | 3,993              | 3,656              |
| Actual Positive   | 753                | 2,074              |

---

## Project Structure

```
multi_diseases/
│
├── model.ipynb                          # Main notebook — training, calibration, evaluation
├── mimic_multidisease_ehr.csv           # Preprocessed MIMIC-IV EHR dataset (52,378 patients)
│
├── threshold_calibration_results.csv    # Per-disease optimal thresholds and metrics
│
├── Sepsis_Label_threshold_curve.png     # F1 vs. Threshold plot — Sepsis
├── AKI_Label_threshold_curve.png        # F1 vs. Threshold plot — AKI
├── CHF_Label_threshold_curve.png        # F1 vs. Threshold plot — CHF
├── Diabetes_Label_threshold_curve.png   # F1 vs. Threshold plot — Diabetes
│
├── .gitignore                           # Git ignore rules
└── README.md                            # This file
```

---

## Getting Started

### Prerequisites

- **Python** ≥ 3.10
- **Jupyter Notebook** or **JupyterLab**
- Access to the preprocessed MIMIC-IV dataset (`mimic_multidisease_ehr.csv`)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Vishvambar/IEEE_Research_Project.git
   cd IEEE_Research_Project
   ```

2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux / macOS
   # venv\Scripts\activate    # Windows
   ```

3. **Install dependencies:**

   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib
   ```

### Running the Model

1. **Launch Jupyter:**

   ```bash
   jupyter notebook model.ipynb
   ```

2. **Execute all cells** — the notebook will:
   - Load and split the dataset (80/20)
   - Train the multi-output XGBoost model
   - Perform threshold calibration across all four diseases
   - Generate threshold curve plots (`*_threshold_curve.png`)
   - Save calibration results to `threshold_calibration_results.csv`
   - Print detailed classification reports and confusion matrices

---

## Methodology

The complete pipeline follows these steps:

### 1. Data Preprocessing
The MIMIC-IV database is preprocessed into a flat CSV where each row represents a single patient admission. Vital sign features are aggregated into min/max/avg summaries per ICU stay. Binary disease labels are derived from ICD diagnosis codes.

### 2. Multi-Output Classification
Rather than training four independent models, we use `sklearn.multioutput.MultiOutputClassifier` to wrap a single XGBoost configuration. This:
- Ensures consistent hyperparameters across all diseases
- Enables parallel training via `n_jobs=-1`
- Simplifies deployment (one model object, four outputs)

### 3. Probabilistic Prediction
Instead of hard class predictions, the model outputs **calibrated probabilities** for each disease. This enables:
- Flexible threshold tuning post-training
- Risk stratification (e.g., low / medium / high risk tiers)
- Integration with clinical decision support dashboards

### 4. Threshold Calibration
The default 0.5 threshold is inappropriate for imbalanced clinical data. The pipeline performs a grid search over thresholds (0.05–0.95) and selects the threshold that maximizes F1-score per disease — balancing precision and recall.

### 5. Clinical Evaluation
Final evaluation uses the calibrated thresholds to generate:
- Per-class precision, recall, and F1-score
- Overall accuracy
- Confusion matrices for clinical interpretability

---

## Limitations & Future Work

### Current Limitations

- **Static features only** — the model uses aggregated vitals/labs and does not capture temporal dynamics (e.g., trends in creatinine over 48 hours)
- **No external validation** — results are from a single MIMIC-IV cohort; generalizability to other hospital systems is unverified
- **Moderate performance on Diabetes/CHF** — F1 scores of ~0.49 and ~0.51 suggest these conditions may require additional features (e.g., HbA1c, BNP levels)
- **No feature importance analysis** — SHAP or permutation importance could improve interpretability

### Future Directions

- **Temporal modeling** — Incorporate LSTM or Transformer architectures to model time-series vitals
- **Feature expansion** — Add medication history, prior diagnoses, and additional lab panels
- **SHAP explainability** — Generate per-patient feature attribution for clinical transparency
- **External validation** — Test on eICU or institutional datasets
- **Deployment** — Package as a REST API or integrate into clinical EHR dashboards

---

## License

This project is developed for academic research purposes under IEEE publication guidelines.

---

<p align="center">
  <i>Built with ❤️ for advancing clinical decision support through machine learning.</i>
</p>
