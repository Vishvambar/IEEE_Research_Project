Week 1 – Research Analysis & Planning



Project Title : Multi-Disease Prediction System

Objectives

Understand the problem statement clearly

Study existing research work

Identify methodology and tools

Plan project execution

Activities

Study domain background and existing solutions

Perform literature survey using IEEE/research papers

Identify research gaps

Finalize dataset(s)

Understand dataset attributes and preprocessing needs

Select tools, frameworks, and technologies

Design system workflow/architecture

Create implementation timeline

Expected Deliverables

Literature Survey

Research Gap Analysis

Dataset Study Report

Proposed Methodology

System Architecture/Workflow Diagram

Weekly Execution Plan































Research Internship Documentation Templates

1. Literature Survey Template

Sr. No.

Paper Title

Authors

Year

Methodology Used

Key Findings

Limitations

Relevance to Project

1

MIMIC-IV: A Freely Accessible Electronic Health Record Dataset

Johnson et al.

2023

Large-scale ICU Electronic Health Record dataset containing demographics, physiological measurements, laboratory tests, diagnoses, medications, and outcomes.

Provides real-world ICU patient data for developing and validating machine learning models.

Contains missing values, heterogeneous measurements, and requires extensive preprocessing.

Used as the primary real-world dataset for developing and validating the disease prediction models.

2

XGBoost: A Scalable Tree Boosting System





Chen and Guestrin

2016

Gradient boosted decision trees with optimized training and handling of missing values.

Provides state-of-the-art performance on structured clinical datasets.

Model interpretation can be difficult without explainability techniques.

Selected as the primary machine learning algorithm for disease prediction.





3

SHAP (SHapley Additive exPlanations)

Lundberg and Lee





2017

Game-theory based explainable AI framework.

Provides local and global feature importance explanations.

Computationally expensive for large datasets.

Used for interpreting disease prediction results and identifying clinically important variables.

4

Early Prediction of Sepsis from Clinical Data Using Machine Learning



Desautels et al.

2016

Machine learning algorithms trained on ICU physiological measurements and laboratory parameters for early sepsis detection.

Vital signs and laboratory parameters can identify septic patients several hours before clinical diagnosis.

Focused only on sepsis prediction and did not address multiple diseases simultaneously.

Supports the use of physiological features such as heart rate, respiratory rate, blood pressure, and laboratory values for disease prediction.





5

Development and Validation of a Machine Learning Model for Prediction of Hospital Mortality

Rajkomar et al.

2018

Deep learning and machine learning models trained on large-scale electronic health records.

Machine learning significantly improves prediction accuracy compared to traditional clinical scoring systems.



Limited explainability and interpretability for clinicians.

Demonstrates the effectiveness of AI-based predictive analytics in healthcare and motivates the use of explainable AI techniques such as SHAP.



















2. Research Gap Analysis Template



Sr No 

Existing Research/Method

Identified Limitation

Research Gap

Proposed Improvement

1

Single disease prediction systems

Predict only one disease at a time

Lack of simultaneous disease prediction

Develop multi-disease prediction framework

2

Black-box clinical models

Limited interpretability

Clinicians cannot understand predictions

Integrate SHAP explainability

3

Synthetic dataset studies

Limited real-world validation

Poor generalization to actual patients

Validate using MIMIC-IV real ICU data

4

Mortality-focused systems

Ignore disease-level prediction

Limited clinical applicability

Predict Sepsis, AKI, CHF, and Diabetes simultaneously



3. Dataset Study Report Template

Parameter

Details

Dataset Name

MIMIC-IV (Medical Information Mart for Intensive Care)

Source of Dataset

PhysioNet

Domain

Healthcare / Critical Care Medicine

Number of Records

52,378 ICU Patients

Number of Features

19

Target Variable

Sepsis_Label
AKI_Label
CHF_Label
Diabetes_Label

Data Type

Structured Electronic Health Records

Missing Values Present

Yes

Preprocessing Required

Missing value handling

Feature extraction

Label generation from ICD codes

Patient-level deduplication

First ICU stay selection



Tools Used for Analysis

Python

Pandas

NumPy

BigQuery

XGBoost

SHAP

Scikit-learn

Observations

The extracted MIMIC-IV cohort contains 52,378 adult ICU patients with real-world physiological measurements and laboratory observations. Disease prevalence was found to be 15.84% for Sepsis, 29.75% for AKI, 23.84% for CHF, and 27.68% for Diabetes. The dataset demonstrates moderate class imbalance and realistic missingness patterns, making it suitable for evaluating machine learning models under real clinical conditions.



Feature Description Table

Feature Name

Description

Age 

Patient Age 

Gender 

Patient Gender

Heart_rate

Patient Heart Rate

Systolic Blood Pressure

Patients Systolic Blood Pressure 

Resp_rate

Patients Respiratory Rate

SpO2_min

Minimum Oxygen Saturation 

WBC_max

Maximum white blood cells count 











4. Proposed Methodology Template

Step No.

Methodology Step

Description

1

Problem Definition

Develop a machine learning system capable of predicting multiple diseases using ICU patient physiological data.





2

Data Collection

Extract patient information from the MIMIC-IV database using SQL queries.

3

Data Preprocessing

Handle missing values, remove duplicates, and create disease labels.

4

Feature Engineering

Generate statistical features including minimum, maximum, and average physiological measurements.

5

Model Development

Train XGBoost-based multi-disease prediction models.

6

Training & Testing

Apply train-test split and cross-validation techniques.

7

Performance Evaluation

Evaluate using ROC-AUC, PR-AUC, Precision, Recall, and F1-Score.

8

Result Analysis

Use SHAP values to identify clinically important features.

























5. System Architecture / Workflow Diagram Template

















6. Weekly Execution Plan Template

Week

Planned Activities

Expected Outcome

Status/Remarks

Week 1

Literature survey, research gap analysis, MIMIC-IV dataset study, SQL-based data extraction, disease label generation, baseline mode



Research foundation established and baseline multi-disease prediction system created



completed

Week 2

Advanced preprocessing, missing value analysis, feature engineering, hyperparameter tuning, threshold optimization, model comparison studies

Improved prediction performance and clinically optimized disease detection thresholds

Planned

Week 3

Explainability analysis using SHAP, visualization generation, architecture refinement, comparative study between synthetic and real-world datasets

Clinically interpretable prediction framework with validated results

Planned

Week 4

Final model validation, dashboard development, documentation, report preparation, presentation creation, result compilation

Complete research project ready for submission and presentation

Planned



Additional Progress Tracking Template

Date

Work Completed

Challenges Faced

Next Plan

Week 1

Literature survey, dataset extraction, model development, benchmarking, threshold optimization, SHAP analysis

Missing clinical values, ICD code mapping, class imbalance

Improve model performance and perform explainability analysis

Week 2 

Feature engineering and optimization

To be updated

Model Refinement

Week 3 

Explainability and visualization

To be Updated 

Final validation 

Week 4

Documentation and reporting

To be Updated 

Project Submission