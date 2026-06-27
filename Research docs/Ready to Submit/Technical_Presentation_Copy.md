# Technical Presentation Script & Outline (Simplified Version)
*Use this structure to build your final PowerPoint slides. The language here is simplified so you can present it naturally and confidently.*

---

## Slide 1: Title Slide
*   **Project Title:** Clinivora: Multi-Disease Early Screening Using Deep Multi-Task Learning
*   **Team Members:** Vishvambar Ramesh Udavant, Ish Prafull Chaniyara, Sagar Shahgond
*   **Faculty Guide:** Dr. Ranjana Kale
*   **Affiliation:** MIT ADT University, Pune

---

## Slide 2: The Clinical Problem (Why we built this)
*   **Alarm Fatigue:** In the ICU, monitors are constantly beeping. Most are false alarms because traditional systems look at diseases (Sepsis, AKI, CHF, Diabetes) separately, ignoring how they are connected.
*   **The Chain Reaction:** We know that a patient's baseline health (like having Diabetes) makes their body weaker, which makes them more likely to get an infection (Sepsis), which then damages their kidneys (AKI).
*   *Visual:* `results/Disease_Cascade_Diagram.svg` (Flowchart showing the clinical links between these diseases).
*   **The Issue with Old Models:** If you use separate models (like XGBoost), they only pass a simple "Yes/No" or a single percentage to the next model. They lose all the detailed clinical context.

---

## Slide 3: Our Proposed Solution (The PyTorch DAG)
*   **Causal DAG Architecture:** We built a network using PyTorch structured as a Directed Acyclic Graph (DAG) that flows like a timeline: `Diabetes → CHF → Sepsis → AKI`.
*   **Detailed Context Sharing:** Instead of passing a single "Yes/No" number between diseases, our network passes a **32-dimensional hidden vector** (a bundle of 32 numbers representing the patient's state).
*   **The Stop-Gradient (`.detach()`) Trick:** Normally, the error of a downstream disease (like AKI) would flow backward and ruin the predictions of upstream diseases (like Diabetes). We used `.detach()` to block the backward error flow while still letting the patient context flow forward.
*   *Visual:* `results/Architecture_Diagram.svg` (The clean flow diagram of our model).

---

## Slide 4: Dataset & Experimental Setup
*   **The Data:** 52,378 adult ICU patient records from the real-world **MIMIC-IV** database.
*   **The Features:** 17 simple features, including Age, Gender, and vitals (Heart Rate, Blood Pressure, Respiratory Rate) along with their changes (Min, Max, Average, and Range).
*   **Focal Loss:** Because most ICU patients are healthy and don't have these diseases, standard training fails. We used Focal Loss to force the model to focus on the rare, hard-to-predict sick patients instead of guessing "healthy" for everyone.

---

## Slide 5: Results & Threshold Calibration (Making it safe)
*   **Avoiding False Alarms:** We didn't just accept the default settings. We ran a strict safety check: the model must achieve high accuracy (Precision $\ge$ 0.65) to prevent false alarms, while still catching at least 50% of sick patients (Recall $\ge$ 0.50).
*   **The Final Scores:**
    *   Diabetes: Precision 0.374 (Unviable - Needs more historical data)
    *   CHF: Precision 0.403 (Unviable - Needs more historical data)
    *   Sepsis: Precision 0.411 (Unviable - Needs more historical data)
    *   **AKI: Precision 0.724 (Viable & highly accurate!)**

---

## Slide 6: Result Analysis (XGBoost vs. PyTorch DAG)
*   **Performance Comparison:** On the exact same dataset, the PyTorch DAG model improved AKI prediction precision from **0.700** (XGBoost baseline) to **0.719** (DAG model).
*   **Why did the DAG win?** Because the AKI head was able to look at the 32-D context vector flowing from Sepsis, CHF, and Diabetes, making a smarter decision.
*   **Why did the other diseases fail?** An AI cannot make guesses out of thin air. You cannot predict long-term Diabetes or CHF using only 24 hours of ICU heart rates. You need long-term medical history. We are honest about this limitation.
*   *Visual:* `results/Evolution_Diagram.svg` (Comparison showing why the DAG retains more information than XGBoost).

---

## Slide 7: Explainable AI (Captum)
*   **Opening the Black Box:** Doctors will not trust a model if they don't know *why* it made a prediction. We used **PyTorch Captum** to calculate feature importance.
*   **Top 5 features for AKI:**
    1. Creatinine Max (The standard medical biomarker for kidney health)
    2. Systolic BP Min
    3. Systolic BP Avg
    4. Gender
    5. SpO2 Min
*   **Medical Validation:** The model naturally learned that `Creatinine` and low blood pressure (`BP Min`) are the most important features for kidney failure. This proves our AI is thinking like a real doctor.
*   *Visual:* `results/Captum_Attributions_AKI.png`

---

## Slide 8: The Clinivora Dashboard (Live Demo)
*   **The Web App:** We didn't just build code; we built a working website dashboard.
*   **Frontend:** Built with React, featuring a premium UI, dark mode, and dynamic medical risk gauges.
*   **Backend:** FastAPI serving our trained PyTorch DAG model in real-time.
*   *(At this point, show screenshots of the React UI or run it on localhost).*

---

## Slide 9: Future Work (V2 Roadmap)
*   To make this system fully ready for hospital use, we plan to implement:
    1.  **Multi-Parent DAG:** Connect Diabetes, CHF, and Sepsis in parallel to feed into AKI, representing a more realistic comorbidity structure.
    2.  **Gradient Surgery:** Evaluate advanced algorithms like PCGrad to resolve conflicting task gradients instead of completely blocking them.
    3.  **Graph Learning:** Let the model automatically discover how diseases are linked using Graph Neural Networks (GNNs).
    4.  **Time-Series Modeling:** Move from statistical ranges to raw sequence data using LSTMs or Transformers.

---

## Slide 10: Q&A
*   Thank you! We are open to your questions.
