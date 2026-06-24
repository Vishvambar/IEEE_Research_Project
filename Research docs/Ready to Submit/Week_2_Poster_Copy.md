# IEEE Week 2 Poster Copy

*Use this text to populate the boxes in your IEEE Week 2 Poster Template.pptx*

---

## 1. Top Header Information

**DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING**

**Project Title:** Multi-Disease Early Screening Using Deep Multi-Task Learning

**Students:** 
Vishvambar Ramesh Udavant
Ish Prafull Chaniyara
Sagar Shahgond

**Faculty Guide:** 
Dr. Ranjana Kale

---

## 2. Problem Statement

In the ICU, patients often present with overlapping and co-occurring conditions. Traditional diagnostic models treat diseases independently, failing to capture critical comorbidity interactions. As a result, they suffer from high false-positive rates, leading to severe alarm fatigue for clinicians. There is a critical need for a unified system that mathematically links chronic baseline risks to acute organ failure complications.

---

## 3. Proposed Solution

We developed a **PyTorch Directed Acyclic Graph (DAG) Neural Network** using Multi-Task Learning. 
Instead of outputting independent binary flags, the network passes dense, latent representations of chronic conditions (like Diabetes) directly downstream into the prediction branches for acute conditions (like Sepsis and AKI). To prevent Catastrophic Interference during backpropagation, we instituted a strict **Stop-Gradient** mathematical isolation technique, ensuring upstream models are protected from massive downstream loss gradients.

---

## 4. Scope and Feasibility

- **4 Disease Modules:** Diabetes, Congestive Heart Failure (CHF), Sepsis, and Acute Kidney Injury (AKI).
- **Scale:** Utilizes 52,378 real-world patient records from the MIMIC-IV clinical database.
- **Explainability:** Replaced black-box processing with PyTorch-native **Captum Integrated Gradients** to mathematically prove to physicians which of the 17 physiological features drove the final prediction.
- **Clinical Safety Net:** Enforces a rigid threshold logic of Precision $\ge$ 0.65 and Recall $\ge$ 0.50.

---

## 5. Preliminary Results (Bullet Points for Results Box)

- **Causal DAG Success:** The shared trunk representations allowed the AKI complication model to reach high clinical viability (Precision 0.724, Recall 0.718).
- **Scientific Rigor:** Identified that current 17 baseline features lack the historical EHR signal required to accurately predict Diabetes and CHF without triggering false alarms, officially designating them as "Clinically Unviable" in V1 and establishing clear parameters for V2.

---

## 6. Proposed Architecture/Diagram (Text to reconstruct the flowchart)

*(You can use shapes in PowerPoint to draw this based on this text flow)*

[ 17 Physiological Features ] 
        ↓
[ Shared Neural Encoder Trunk ]
        ↓
[ Diabetes Head ] → (Passes 32-D Hidden State + Stop Gradient)
        ↓
[ CHF Head ] → (Passes 32-D Hidden State + Stop Gradient)
        ↓
[ Sepsis Head ] → (Passes 32-D Hidden State + Stop Gradient)
        ↓
[ AKI Head ] → Final Output Prediction
