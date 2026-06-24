# Technical Presentation Script & Outline
*Use this structure to build your final PowerPoint presentation.*

## Slide 1: Title Slide
- **Title:** Clinivora: Multi-Disease Early Screening Using Deep Multi-Task Learning
- **Team:** Vishvambar Ramesh Udavant, Ish Prafull Chaniyara, Sagar Shahgond
- **Guide:** Dr. Ranjana Kale

## Slide 2: The Clinical Problem
- **Alarm Fatigue:** Modern ICUs suffer from massive false-positive rates. Traditional systems treat overlapping diseases (Sepsis, AKI, CHF, Diabetes) independently.
- **The "Multi-Task Illusion":** Using disjoint models (like XGBoost Classifier Chains) ignores the comorbidity cascade, failing to understand that a patient's baseline metabolic risk dictates their reaction to an acute infection.

## Slide 3: The Proposed Solution
- **Causal DAG Architecture:** We built a custom PyTorch Directed Acyclic Graph (DAG) that passes *dense latent representations* of chronic conditions downstream into acute condition branches.
- **Stop-Gradient Backpropagation:** We isolated the error loss of acute conditions (like AKI) from rewriting the weights of chronic conditions (like Diabetes) using mathematical tensor detachment.

## Slide 4: Experimental Setup & Dataset
- **Dataset:** 52,378 ICU patient records extracted from the MIMIC-IV database.
- **Features:** 17 physiological parameters (Demographics, Labs, and Vital Sign Volatility ranges).
- **Optimization:** Implemented a Static Focal Loss ($\gamma=2$) to combat massive class imbalance and force the network to learn hard positive examples.

## Slide 5: Threshold Calibration (The Reality Check)
- **The Standard:** We enforced a strict clinical safety net: `Recall >= 0.50` and `Precision >= 0.65`.
- **Results Table:**
  - Diabetes: Precision 0.374 (Unviable)
  - CHF: Precision 0.403 (Unviable)
  - Sepsis: Precision 0.411 (Unviable)
  - **AKI: Precision 0.724 (Viable)**

## Slide 6: Result Analysis
- **Why did AKI succeed?** Because it received the dense latent context from the upstream models, proving the power of the DAG architecture.
- **Why did the others fail?** A neural network cannot invent signal from noise. The 17 core features fundamentally lack the historical EHR data (like prior HbA1c labs) required to confidently predict chronic conditions. 
- *Insert `EDA_correlation_heatmap.png` here.*

## Slide 7: Captum Explainable AI (XAI)
- Deep learning is a "black box." We replaced TreeSHAP with **PyTorch Captum Integrated Gradients**.
- This mathematically integrates the path from a baseline tensor to the patient's actual vitals, proving to clinicians exactly which physiological features drove the AKI prediction.
- *Insert `Captum_Attributions_AKI.png` here.*

## Slide 8: Live Product Demo
- **Clinivora Dashboard:** A full-stack web application.
- **Frontend:** React (Vite) with a premium dark/light mode UI and dynamic risk meters.
- **Backend:** FastAPI (Python) serving the PyTorch DAG model in real-time.
- *(Switch screen to show the working localhost web app, or embed screenshots of the UI here).*

## Slide 9: Conclusion & Future Work
- We successfully engineered a deep learning pipeline that respects the clinical timeline.
- **V2 Roadmap:** Integration of sequential time-series data (RNN/Transformers) and historical lab results to cross the viability threshold for Sepsis, CHF, and Diabetes.

## Slide 10: Q&A
- Thank you! Any questions?
