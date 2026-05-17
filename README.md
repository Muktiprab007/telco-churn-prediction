# 📉 Telco Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-2980B9?style=for-the-badge&logo=lightgbm&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00?style=for-the-badge&logo=catboost&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)

A comparative study of three gradient boosting algorithms — **LightGBM**, **XGBoost**, and **CatBoost** — for predicting customer churn on the Telco Customer dataset (7,043 records, 50 features).

---

## 📌 Project Overview

Customer churn is one of the most critical challenges in the telecommunications industry. This project builds an end-to-end machine learning pipeline to identify customers at risk of churning, enabling businesses to take proactive retention actions.

---

## 📂 Dataset

- **Source:** [Telco Customer Churn (11.1.3+) — Kaggle](https://www.kaggle.com/datasets/alfathterry/telco-customer-churn-11-1-3)
- **Records:** 7,043 customers
- **Features:** 50 columns (demographics, services, contract, billing)
- **Target:** `Churn Label` (Yes / No) — imbalanced at ~26% churn rate

---

## 🔍 Exploratory Data Analysis

Key findings from EDA:

- **46%** of customers on monthly contracts churn vs **3%** on two-year contracts
- **Fiber Optic** users have the highest churn rate among internet types
- **Satisfaction Score** is the strongest numerical predictor of churn (correlation: -0.75)
- **Competitor offers** are the leading churn reason, accounting for 45% of churned customers

---

## ⚙️ Workflow

```
Data Loading → EDA → Preprocessing → Modeling → Evaluation
```

### Preprocessing
- Dropped identifier and data leakage columns (`Churn Score`, `Churn Reason`, `Customer Status`, etc.)
- Handled missing values: `Internet Type` → `'None'`, `Offer` → `'No Offer'`
- Encoded categorical features using Label Encoding
- Applied **class weight balancing** to handle imbalanced target labels

### Modeling
Three gradient boosting models were trained and compared:

| Model | Class Imbalance Handling |
|---|---|
| LightGBM | `class_weight='balanced'` |
| XGBoost | `scale_pos_weight` (auto-calculated) |
| CatBoost | `auto_class_weights='Balanced'` |

---

## 📊 Results

| Model | ROC-AUC |
|---|---|
| LightGBM | 0.9917 |
| XGBoost | 0.9901 |
| **CatBoost** | **0.9927** ✅ |

**CatBoost** achieved the highest ROC-AUC of **0.9927**, making it the best-performing model on this dataset.

---

## 📁 Repository Structure

```
telco-churn-prediction/
│
├── telco-churn-lgbm-xgb-catboost-comparison.ipynb   # Main notebook
└── README.md
```

---

## 🛠️ Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
lightgbm
xgboost
catboost
```

---

## 👤 Author

**Mukti Prabowo**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/muktiprabowo)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/muktiprab007)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/Muktiprab007)
