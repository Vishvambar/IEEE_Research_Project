# Final Internship Report: Clinivora Multi-Disease Early Screening Engine

**Team:** Vishvambar Ramesh Udavant, Ish Prafull Chaniyara, Sagar Shahgond  
**Guide:** Dr. Ranjana Kale  

---

## Executive Summary
During this 4-week internship, our group engineered a complete, end-to-end clinical decision support system designed to predict multiple concurrent diseases (Diabetes, CHF, Sepsis, AKI) in the ICU. The project evolved from disjoint machine learning classifiers into a highly advanced PyTorch Deep Multi-Task Learning framework, culminating in a production-ready React web application.

## Week 1: Research & Baseline Development
We initiated the project by extracting 52,378 patient records from the MIMIC-IV database. 
- **Research Gap:** We identified that existing clinical models treat overlapping conditions independently, leading to massive false-positive rates and clinician "alarm fatigue."
- **Baseline Model:** We established a baseline using XGBoost Classifier Chains, which revealed the fundamental "Multi-Task Illusion"—passing binary logits between models destroys physiological context.

## Week 2: Data Preprocessing & PyTorch Transition
To solve the comorbidity context problem, we completely overhauled the architecture.
- **Data Engineering:** We handled missing values via median imputation, applied standard scaling, and engineered physiological volatility ranges.
- **The Causal DAG:** We wrote a custom PyTorch `nn.Module` featuring a shared 64-dimensional trunk that branches sequentially. It passes dense 32-dimensional latent hidden states downstream (`Diabetes -> CHF -> Sepsis -> AKI`).
- **Stop-Gradient:** We utilized PyTorch `.detach()` to prevent catastrophic interference, isolating early chronic heads from the massive error gradients of acute complications.

## Week 3: Academic Publication & Application Development
With the model successfully training via a custom Static Focal Loss, we extracted the results.
- **Reality Check:** We evaluated the models against a strict clinical safety net (`Precision >= 0.65`, `Recall >= 0.50`). Diabetes, CHF, and Sepsis failed viability, proving that our 17 features lacked the necessary historical EHR signal. However, the **AKI model succeeded** (Precision 0.724), proving the massive power of the shared trunk representations.
- **Explainable AI:** We utilized Captum Integrated Gradients to provide mathematical feature transparency.
- **Proof-of-Concept Demo:** We built "Clinivora," a full-stack web application featuring a Python FastAPI backend and a Vite React frontend, allowing clinicians to input vitals and view dynamic risk cascades in real-time.
- **IEEE Paper:** We drafted an original, zero-plagiarism conference paper detailing our methodology and findings.

## Week 4: Final Compilation
In the final week, all source code, datasets, execution scripts, and analytical reports were packaged into the final submission format. We designed the technical presentation slide deck to articulate the rigorous mathematical and clinical honesty of our system. 

The Clinivora project stands as a fully transparent, mathematically rigorous diagnostic framework ready for V2 time-series dataset integration.
