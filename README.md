# TATA Steel Machine Failure Prediction

## 📌 Project Overview

A machine learning-based predictive maintenance solution designed to predict potential machine failures using operational and sensor parameters.

## 🎯 Business Objective

The objective is to identify potential machine failures early and support proactive maintenance, reducing unexpected downtime and improving operational efficiency.

## 📊 Dataset

- 136K+ machine records
- Temperature
- Rotational Speed
- Torque
- Tool Wear
- Machine Failure indicators

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- LightGBM
- Matplotlib
- Seaborn
- Streamlit

## 🤖 Machine Learning

Models evaluated:
- Logistic Regression
- Random Forest
- XGBoost

Model optimization included:
- 5-Fold Cross-Validation
- GridSearchCV / RandomizedSearchCV
- Class Weighting
- ROC-AUC
- PR-AUC

## 🚀 Deployment

The optimized XGBoost model was deployed using Streamlit to provide real-time machine failure predictions, probability scores, and maintenance alerts.

## 📷 Dashboard

![Dashboard](images/dashboard.png)

## 📈 Key Outcome

Developed an interactive predictive-maintenance application capable of analyzing machine parameters and identifying potential failure risks.
