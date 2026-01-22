# Loan Default Prediction Using Machine Learning

## Project Overview
This project aims to predict loan defaults in peer-to-peer (P2P) lending using supervised machine learning techniques. It employs Logistic Regression, Random Forest Classifier, and XGBoost Classifier to estimate the probability that a borrower will default within 12 months of loan issuance.  

By accurately predicting defaults, the project supports financial platforms and investors in risk assessment, improves transparency, and enhances decision-making.

---

## Table of Contents
1. [Problem Definition](#problem-definition)
2. [Objectives](#objectives)
3. [Methodology](#methodology)
4. [Feature Engineering](#feature-engineering)
5. [Algorithms](#algorithms)
6. [Experimental Design](#experimental-design)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Results & Interpretation](#results--interpretation)
9. [Model Comparison](#model-comparison)
10. [Feature Importance & SHAP Analysis](#feature-importance--shap-analysis)
11. [Limitations](#limitations)
12. [Conclusion](#conclusion)
13. [References](#references)

---

## Problem Definition
P2P lending platforms allow individuals to borrow and invest directly without traditional financial intermediaries. While this increases accessibility, it also raises the risk of borrowers defaulting.  

This project focuses on **predicting whether a borrower will default within 12 months**, helping investors and platforms mitigate potential losses.

---

## Objectives
- Predict loan defaults using historical loan data from Bondora.
- Compare performance of Logistic Regression, Random Forest, and XGBoost classifiers.
- Identify key factors contributing to loan defaults.
- Apply explainable AI methods (SHAP) to improve model transparency.

---

## Methodology

### Data Acquisition
- Dataset: Bondora P2P Loan Data (publicly available)
- Environment: Python with Jupyter Notebook
- Data: 644,037 rows, 31 columns (categorical and numerical features)

### Data Preprocessing
- Dropped irrelevant columns to prevent data leakage.
- Converted date-time features into numeric values.
- Imputed missing values using median (numeric) and mode (categorical).
- Separated features (X) and target variable (y).

### Exploratory Data Analysis (EDA)
- Inspected distributions of numerical and categorical variables.
- Analyzed correlations and missing values.
- Visualized feature statistics.

---

## Feature Engineering
- **Feature Selection:** Only features known at loan issuance were used.
- **Feature Transformation:** Separated numerical and categorical features.
- **Feature Scaling:** StandardScaler for numeric columns, OneHotEncoder for categorical columns.
- Final dataset was fully numeric and normalized.

---

## Algorithms

1. **Logistic Regression:** Simple, transparent, computationally efficient; predicts binary outcomes.  
2. **Random Forest Classifier:** Ensemble method using multiple decision trees; robust and handles complex relationships.  
3. **XGBoost Classifier:** Gradient boosting algorithm; handles large datasets, regularization prevents overfitting.

---

## Experimental Design

### Train-Test Split
- 70% training, 30% testing

### Handling Class Imbalance
- SMOTE (Synthetic Minority Oversampling Technique) applied to training set to improve recall for default cases.

### Model Training
- Pipelines created for preprocessing, scaling, and training.
- Threshold tuning applied for Logistic Regression.

---

## Evaluation Metrics
- **Accuracy:** Correct predictions overall
- **Precision:** Correct default predictions
- **Recall:** Captures minority class (defaults)
- **F1-Score:** Balance of precision and recall
- **ROC-AUC:** Discrimination ability
- **Confusion Matrix:** True/False positives/negatives

---

## Results & Interpretation

- **Logistic Regression:** Accuracy 0.83, Recall for defaults low (0.46), struggles with imbalanced data.  
- **Random Forest:** Stronger performance, captures complex relationships, higher recall and precision.  
- **XGBoost:** Highest recall, identifies most risky borrowers, slightly lower precision, best overall for risk minimization.

---

## Model Comparison

| Model                  | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score | ROC-AUC |
|------------------------|----------|--------------------|-----------------|----------|---------|
| Logistic Regression    | 0.83     | 0.46               | 0.34            | 0.39     | 0.71    |
| Random Forest          | 0.87     | 0.63               | 0.72            | 0.67     | 0.82    |
| XGBoost                | 0.86     | 0.58               | 0.82            | 0.68     | 0.84    |

---

## Feature Importance & SHAP Analysis

- Key contributors: **Country, Initial Interest Rate, Customer Risk Rating, Number of Payments, Loan Year, Income**  
- SHAP plots highlight the influence of these features on model predictions.

---

## Limitations
- Class imbalance affects model performance despite SMOTE.
- Random Forest and XGBoost require higher computational resources.
- SHAP improves interpretability but ensemble models remain complex.
- Missing values and data inconsistencies may affect model behavior.

---

## Conclusion
- XGBoost provided the best performance for predicting defaults.
- Random Forest is also reliable and interpretable.
- Logistic Regression offers moderate results but is valuable for coefficient-based insights.
- Feature importance and SHAP analysis enhance transparency and guide lending decisions.

---

## References
1. Belcic, I. and Stryker, C. (2021). Supervised Learning. IBM. [Link](https://www.ibm.com/think/topics/supervised-learning)  
2. Ko, P.-C. et al. (2022). P2P Lending Default Prediction Based on AI and Statistical Models. *Entropy*, 24(6), p.801.  
3. rohanc7 (2021). Bondora Peer to Peer Lending. [Kaggle](https://www.kaggle.com/code/rohanc7/bondora-peer-to-peer-lending/notebook#Preparing-data-for-Modelling)  
4. OpenAI (2025). ChatGPT (GPT-5.1). [Link](https://chat.openai.com/)  
5. Other references 

---

