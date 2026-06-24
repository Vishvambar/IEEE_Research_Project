# Multi-Disease Early Screening Using PyTorch Multi-Task Learning on MIMIC-IV EHR Data

> **IEEE Research Project** — A deep multi-task learning clinical decision support system for simultaneous early detection of Sepsis, Acute Kidney Injury (AKI), Congestive Heart Failure (CHF), and Diabetes from Electronic Health Records.

---

## Table of Contents

- [Overview](#overview)
- [Clinical Motivation](#clinical-motivation)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Model Architecture (PyTorch DAG)](#model-architecture-pytorch-dag)
- [Threshold Calibration](#threshold-calibration)
- [Explainability (Captum)](#explainability-captum)
- [Results](#results)
  - [Threshold Optimization Summary](#threshold-optimization-summary)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Limitations & Future Work](#limitations--future-work)

---

## Overview

This project presents a **PyTorch Directed Acyclic Graph (DAG) Neural Network** designed to simultaneously predict the risk of four critical diseases from a single patient encounter using structured EHR data derived from the **MIMIC-IV** clinical database.

Unlike disjoint models (like XGBoost Classifier Chains) that suffer from massive information loss by passing binary 1D logits, this system learns a **Shared Physiological Trunk** and passes **dense, 32-dimensional latent context** down the clinical timeline.

The pipeline includes:
1. **PyTorch DAG Architecture** with causal disease progression.
2. **Stop-Gradient Backpropagation** to isolate early chronic heads from acute wrecking-ball gradients.
3. **Static Focal Loss** scaled by inverse class frequency to aggressively penalize hard positive cases.
4. **Captum Integrated Gradients** for PyTorch-native clinical explainability.

---

## Clinical Motivation

In critical care settings, patients often present with overlapping and co-occurring conditions. A patient’s baseline metabolic and cardiovascular risk heavily dictates how they will react to ICU stress, which can trigger acute complications like Sepsis and subsequent organ failures.

By passing the dense hidden state of chronic predictors (like Diabetes) directly into the branches of acute predictors (like AKI), the neural network intrinsically models comorbidities.

---

## Dataset

| Property               | Value                                      |
|------------------------|---------------------------------------------|
| **Source**             | MIMIC-IV Clinical Database                  |
| **File**               | `mimic_multidisease_ehr.csv`               |
| **Total Patients**     | 52,378                                     |
| **Target Distribution**| Sepsis (15.9%), AKI (29.8%), CHF (22.8%), Diab (27.0%) |

---

## Feature Engineering

The model utilizes **17 clinically interpretable features**:
- 4 Demographics & Lab Values (Age, Gender, wbc_max, creatinine_max)
- 13 Vital Signs & Volatility features

*Volatility Features*: `heart_rate_range`, `sys_bp_range`, `resp_rate_range`.
> **Limitation Acknowledgment:** While range captures volatility, it inherently fails to capture sequence directionality. True sequence modeling requires raw un-aggregated time-series data.

---

## Model Architecture (PyTorch DAG)

The architecture utilizes a shared representation encoder branching into a DAG structure that strictly respects the clinical arrow of time:
1. **Diabetes** (Chronic Metabolic)
2. **CHF** (Chronic/Progressive Cardiovascular)
3. **Sepsis** (Acute Infection)
4. **AKI** (Acute Secondary Complication)

```text
Input: 17 Features
       │
       ▼
[ Shared Encoder Trunk (Dense 64-Dim) ] ───► (Independent gradients from all 4 heads)
       │
       ├────────────────────────────────┐
       ▼                                │
[ Diabetes Head ] ──► P(Diab)           │
       │                                │
 (32-D Hidden State)                    │
  [ .detach() ]                         │
       │                                │
       ▼                                │
[ CHF Head ] ◄──────────────────────────┤
       │                                │
 (32-D Hidden State)                    │
  [ .detach() ]                         │
       │                                │
       ▼                                │
[ Sepsis Head ] ◄───────────────────────┤
       │                                │
 (32-D Hidden State)                    │
  [ .detach() ]                         │
       │                                │
       ▼                                │
[ AKI Head ] ◄──────────────────────────┘
       │
       ▼
     P(AKI)
```

**The Stop-Gradient Mathematical Solution**: 
To prevent Catastrophic Interference, where the massive loss gradient of an acute condition like AKI flows backward and destroys the weights of the Diabetes head, we use `.detach()` on the lateral hidden states. The downstream model treats the upstream state as a frozen informational prior, protecting upstream networks while the shared trunk optimizes the joint physiological embedding.

---

## Threshold Calibration

Our calibration loop enforces the following logic per-disease to combat alarm fatigue:
1. **Primary Goal:** Maximize Recall, subject to **Precision $\ge$ 0.65**.
2. **Clinical Safety Fallback:** Maximize Precision subject to **Recall $\ge$ 0.50**.

If a model triggers the fallback and still yields poor precision, it is branded **Clinically Unviable**.

---

## Explainability (Captum)

For a PyTorch neural network, we utilize `captum.attr.IntegratedGradients`. Captum mathematically integrates the gradient along the path from a zero-baseline to the patient's specific features, cleanly tracing through the DAG and shared trunk.

![Captum Attributions](Captum_Attributions_AKI.png)

---

## Results

### Threshold Optimization Summary (PyTorch Execution)

| Disease        | Optimal Threshold | Precision | Recall | Status |
|---------------|-------------------|-----------|--------|--------|
| **Diabetes**   | 0.50              | 0.374     | 0.674  | ❌ Unviable |
| **CHF**        | 0.50              | 0.403     | 0.667  | ❌ Unviable |
| **Sepsis**     | 0.55              | 0.411     | 0.578  | ❌ Unviable |
| **AKI**        | 0.55              | 0.724     | 0.718  | ✅ Viable |

*Reality Check: The deep learning architecture vastly improved the viable AKI model by supplying dense downstream context. However, a neural network cannot invent signal from noise. Sepsis, CHF, and Diabetes failed to capture 50% of true cases without triggering high rates of false alarms because the 17 core features fundamentally lack the historical data required to predict them accurately.*

---

## Project Structure

```
multi_diseases/
├── mtl_model.py                         # PyTorch nn.Module (DAG Architecture)
├── train_mtl.py                         # Training, Focal Loss, Captum Explainability
├── mimic_multidisease_ehr.csv           # Preprocessed EHR dataset
├── Captum_Attributions_AKI.png          # Native PyTorch IntegratedGradients feature plot
└── README.md                            # Documentation
```

---

## Getting Started

### Installation
```bash
python -m venv venv
source venv/bin/activate
pip install torch captum scikit-learn pandas numpy matplotlib
```

### Running the Model
```bash
python train_mtl.py
```

---

## Limitations & Future Work

1. **Unviable Disease Heads:** As proven by our rigid threshold constraints, Diabetes, CHF, and Sepsis cannot currently be predicted with clinically acceptable precision/recall ratios.
2. **Missing Historical Data:** V2 of this model *must* incorporate historical EHR data (e.g., prior HbA1c labs, outpatient prescriptions) to give the network the necessary signal.
3. **Sequential Time-Series:** The shared trunk requires raw sequential vital sign timestamps to deploy LSTM/Transformer architectures, moving beyond aggregated ranges.
