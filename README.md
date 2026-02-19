# Predictive Maintenance Using Machine Learning

## 📌 Overview

This project implements a supervised machine learning approach for predictive maintenance using industrial sensor data. The objective is to detect rare equipment failure events under severe class imbalance conditions and evaluate the performance of different classification models.

The study compares two machine learning models:

- **Logistic Regression** (interpretable linear model)
- **Random Forest** (ensemble-based nonlinear model)

The project demonstrates how structured data preprocessing, model training, and imbalance-aware evaluation strategies can be applied to real-world industrial maintenance scenarios.

---

## 🎯 Research Objective

The central research question addressed in this project is:

> Can supervised machine learning models effectively predict rare equipment failures using multivariate sensor measurements, and how do different modeling approaches compare under extreme class imbalance?

---

## 📊 Dataset Description

The dataset contains **124,494 observations** collected from industrial equipment and includes:

- Timestamp  
- Device identifier  
- Nine numerical sensor metrics (metric1–metric9)  
- Binary failure label  

Only **106 observations correspond to failure events**, resulting in a highly imbalanced dataset (<0.1% failure rate).

To prevent data leakage, non-predictive attributes such as timestamp and device ID are removed before model training.

---

## ⚙️ Methodology

The machine learning pipeline follows a structured workflow:

1. Data loading and inspection  
2. Removal of non-predictive identifiers  
3. Feature–target separation  
4. Stratified train–test split (75% training / 25% testing)  
5. Feature scaling (Logistic Regression only)  
6. Model training with class-weight balancing  
7. Performance evaluation  

Class imbalance is handled using `class_weight='balanced'` rather than resampling techniques.

---

## 📈 Evaluation Metrics

Because the dataset is highly imbalanced, traditional accuracy is not sufficient.

The models are evaluated using:

- Precision  
- Recall  
- F1-score  
- Confusion Matrix  
- ROC Curve  
- ROC-AUC  

Special emphasis is placed on **recall** and **ROC-AUC**, as failure detection is more critical than overall classification accuracy.

---

## 🏆 Results Summary

- Logistic Regression achieved a ROC-AUC of approximately **0.76**
- Random Forest achieved a ROC-AUC of approximately **0.68**
- Logistic Regression demonstrated superior recall for failure detection

The results indicate that increased model complexity does not necessarily guarantee better performance in highly imbalanced predictive maintenance tasks.

---

## 🛠 Technologies Used

- Python  
- pandas  
- numpy  
- scikit-learn  
- matplotlib  

## 📂 Project Structure
```text
Predictive-Maintenance-ML/
│
├── data/
│ └── predictive_maintenance_dataset.csv
│
├── src/
│ └── predictive_maintenance.py
│
├── figures/
│ ├── roc_curve.png
│ ├── feature_importance.png
│ └── methodology_pipeline.png
│
├── requirements.txt
└── README.md
