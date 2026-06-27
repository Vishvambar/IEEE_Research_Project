# Clinivora: Multi-Disease Early Screening Engine

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![MIMIC-IV](https://img.shields.io/badge/Dataset-MIMIC--IV-blue?style=for-the-badge)
![IEEE Research](https://img.shields.io/badge/IEEE-Research_Project-00629B?style=for-the-badge)

> **An IEEE Summer Research Internship Project**
> 
> A deep Multi-Task Learning (MTL) clinical decision support system designed for the simultaneous early detection of **Sepsis, Acute Kidney Injury (AKI), Congestive Heart Failure (CHF), and Diabetes** from structured Electronic Health Records (EHR).

---

## 📑 Table of Contents
1. [Overview & Clinical Motivation](#-overview--clinical-motivation)
2. [Dataset & Features](#-dataset--features)
3. [Deep DAG Architecture](#-deep-dag-architecture)
4. [Mathematical Solutions](#-mathematical-solutions)
5. [Experimental Results](#-experimental-results)
6. [Explainable AI (XAI)](#-explainable-ai-xai)
7. [Installation & Usage](#-installation--usage)
8. [Project Structure](#-project-structure)
9. [Team & Acknowledgments](#-team--acknowledgments)

---

## 🏥 Overview & Clinical Motivation

In modern Intensive Care Units (ICUs), patients generate thousands of data points per hour. Traditional clinical diagnostic models (like disjoint XGBoost or Random Forest classifiers) treat concurrent diseases independently. This isolated modeling approach creates a **"Multi-Task Illusion"**—it fundamentally fails to capture the clinical comorbidity cascade (e.g., a patient’s baseline Diabetes dictates their resilience during a Sepsis infection, which subsequently drives AKI). 

By ignoring the clinical arrow of time, standard models suffer from poor precision. In a hospital, low precision directly translates to high false-positive rates, causing **"Alarm Fatigue,"** a psychological phenomenon where clinicians become desensitized to safety alerts.

**Clinivora solves this** by utilizing a **PyTorch Directed Acyclic Graph (DAG)**. Instead of passing binary logits between diseases, the neural network intrinsically models comorbidities by passing dense, 32-dimensional latent representations of chronic conditions directly into the predictive branches of acute conditions.

---

## 📊 Dataset & Features

This project utilizes the **Medical Information Mart for Intensive Care (MIMIC-IV v2.2)** database, a comprehensive, de-identified clinical dataset.

| Dataset Metric | Value |
|----------------|-------|
| **Total Cohort Size** | 52,378 Adult ICU Patients |
| **Sepsis Prevalence** | 15.84% |
| **AKI Prevalence** | 29.75% |
| **CHF Prevalence** | 23.84% |
| **Diabetes Prevalence** | 27.68% |

### Core Feature Vector (17-D)
To ensure high clinical interpretability and mitigate data noise, we engineered 17 core features:
* **Demographics**: `Age`, `Gender`
* **Metabolic/Renal Markers**: `WBC Max`, `Creatinine Max`
* **Oxygenation**: `SpO2 Min`
* **Hemodynamic Volatility**: Min, Max, Average, and Range (Max-Min) for:
  * *Heart Rate*
  * *Systolic Blood Pressure*
  * *Respiratory Rate*

*(Note: Volatility metrics like `heart_rate_range` were prioritized as they are powerful indicators of homeostatic instability in critical care).*

---

## 🧠 Deep DAG Architecture

The architecture utilizes a shared representation encoder branching into a DAG structure that strictly respects the clinical arrow of time.

```text
Input: 17 Features
       │
       ▼
[ Shared Encoder Trunk (Dense 64-Dim) ] ───► (Optimized by all 4 disease heads)
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

---

## 📐 Mathematical Solutions

### 1. The Stop-Gradient Solution
A fundamental mathematical flaw in deep DAG architectures is **gradient hijacking** (Catastrophic Interference). During backpropagation, the massive error gradients of an acute condition like AKI flow backward through the entire graph, overwriting the weights of the upstream Diabetes head to serve the AKI objective.

To solve this, we applied the **Stop-Gradient** technique using `tensor.detach()` on the lateral hidden states between diseases. By severing the computational graph laterally, the downstream model treats the upstream state as a fixed informational prior, allowing independent optimization at each head while the shared trunk updates jointly.

### 2. Static Focal Loss
Due to inherent medical class imbalances, standard Binary Cross Entropy (BCE) fails. We implemented **Focal Loss ($\gamma=2.0$)** to dynamically scale the cross-entropy loss, aggressively down-weighting easy negative examples and forcing the optimizer to focus on rare positive cases.

---

## 📈 Experimental Results

### Baseline vs. DAG (AKI Prediction)
We rigorously benchmarked the DAG architecture against classical Logistic Regression and XGBoost Classifier Chains. To combat alarm fatigue, we implemented a strict threshold calibration loop optimizing for **Precision**.

| Disease | Model | ROC-AUC | PR-AUC | Precision | Recall |
|---------|-------|---------|--------|-----------|--------|
| **AKI** | XGBoost Chain | 0.884 | 0.768 | 0.700 | 0.753 |
| **AKI** | **PyTorch DAG** | **0.885** | **0.769** | **0.719** | 0.729 |

*Finding: The DAG's dense contextual sharing resulted in a modest but consistent improvement in downstream AKI precision compared to passing hard 1D binary probability logits.*

### The Reality of Missing Signal
Our rigorously honest evaluation revealed that the 17 core physiological features lack the required historical EHR signal to predict chronic diseases (Diabetes, CHF) with clinical viability. Without prior HbA1c labs or historical ejection fractions, the precision for chronic conditions capped at ~0.40. The DAG architecture cannot invent signal from noise, strictly establishing the boundary limits of current ICU features.

---

## 🔍 Explainable AI (XAI)

Deep learning models in healthcare cannot operate as black boxes. We utilized PyTorch's **Captum Integrated Gradients** to trace feature attributions from the shared trunk directly to the AKI output.

**Top 5 AKI Indicators:**
1. Creatinine Max
2. Systolic BP Min
3. Systolic BP Avg
4. Gender
5. SpO2 Min

The model's heavy reliance on `Creatinine Max` completely aligns with established nephrology literature, mathematically proving the clinical soundness of the learned representations.

---

## ⚙️ Installation & Usage

### Prerequisites
* Python 3.9+
* pip

### Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/yourusername/multi-disease-prediction.git
cd multi-disease-prediction
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
*(If `requirements.txt` is missing, manually install: `pip install torch captum scikit-learn pandas numpy matplotlib seaborn`)*

### Execution
Run the core training and evaluation pipeline:
```bash
python scripts/train_mtl.py
```
This script will automatically:
1. Load the MIMIC-IV cohort.
2. Train the Multi-Task DAG architecture.
3. Perform threshold calibration.
4. Output AUC, Precision, and Recall metrics.
5. Generate the Captum Integrated Gradients explanation plots.

---

## 📂 Project Structure

```text
multi_diseases/
├── data/
│   └── mimic_multidisease_ehr.csv         # Raw/Preprocessed MIMIC-IV Dataset
├── models/
│   ├── mtl_model.py                       # PyTorch nn.Module (DAG Architecture)
│   └── train_mtl.py                       # Core training, loss, and XAI script
├── results/
│   ├── Architecture_Diagram.svg           # High-res DAG flow diagram
│   ├── AKI_Label_threshold_curve.png      # Precision/Recall optimization curve
│   └── Captum_Attributions_AKI.png        # Feature importance plots
├── docs/                                  # Research reports and IEEE drafts
└── README.md                              # This file
```

---

## 👥 Team & Acknowledgments

**Group HC-4 | TY-AIA | MIT ADT University**
* Vishvambar Udavant
* Ish Chaniyara
* Sagar Shahgond

**Faculty Guide:** Dr. Ranjana Kale

*Special thanks to the MIT ADT University Department of Computer Science & Engineering and the IEEE Computational Intelligence Society (Pune Chapter) for supporting this Summer Research Internship.*
