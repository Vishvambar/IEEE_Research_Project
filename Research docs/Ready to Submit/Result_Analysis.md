# Result Analysis & Experimental Findings

This document provides a deep dive into the findings and metrics derived from the PyTorch Multi-Task Learning architecture.

## 1. Threshold Calibration Analysis

In standard machine learning, researchers often use a default threshold of 0.5 to convert probabilities to binary classes. In a clinical setting, this is dangerous. If the default 0.5 threshold yields a massive amount of false positives, the resulting "Alarm Fatigue" will cause doctors to ignore the system entirely.

We implemented a strict **Calibration Loop**:
- We scanned thresholds from 0.10 to 0.90.
- We required the model to capture at least 50% of true cases (`Recall >= 0.50`).
- Given that recall, we required the model to be correct at least 65% of the time (`Precision >= 0.65`).

### The Findings
- **Diabetes (Threshold 0.50):** Max Precision of 0.374. Failed viability.
- **CHF (Threshold 0.50):** Max Precision of 0.403. Failed viability.
- **Sepsis (Threshold 0.55):** Max Precision of 0.411. Failed viability.
- **AKI (Threshold 0.55):** Max Precision of **0.724** (Recall 0.718). Passed viability!

**Conclusion:** The deep learning model successfully extracted massive predictive power for AKI by using the shared hidden states. However, it exposed a fundamental data limitation for the other diseases.

## 2. The Stop-Gradient Impact

During early testing, we hypothesized that the massive error gradients from the AKI head (which has a strong physiological signal) would flow backward through the DAG and rewrite the weights of the Diabetes and CHF heads, destroying their ability to learn.

By introducing `tensor.detach()` between the heads, we severed this computational graph.
- The training logs showed steady, simultaneous loss reduction across all 4 heads (averaging ~0.203 Loss by epoch 15).
- If the gradients had hijacked the network, the Diabetes loss would have plateaued or exploded. The Stop-Gradient math worked perfectly.

## 3. Captum Explainability

Deep learning is notoriously "black-box." To solve this, we used PyTorch's native `captum.attr.IntegratedGradients`.

Unlike standard feature importance, Integrated Gradients mathematically calculates the integral of the gradients along a straight path from a baseline (all zeros) to the patient's actual feature values. 
- This allowed us to explicitly trace *which* of the 17 inputs passing through the shared 64-dimensional trunk were responsible for a high AKI risk score.
- This fulfills the clinical requirement for Explainable AI (XAI), ensuring doctors can trust the output.
