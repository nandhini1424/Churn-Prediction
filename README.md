# 📊 Customer Churn Prediction with Random Forest

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/status-Completed-brightgreen)

> A machine learning model to predict customer churn using the [Telco Customer Churn dataset](https://www.kaggle.com/blastchar/telco-customer-churn).  
> Built with `RandomForestClassifier` and deployed as an interactive script for real-time predictions.

---

## ✨ Features

- Uses customer data (e.g., contract type, payment method, tenure, services used)
- Applies label encoding to categorical variables
- Trains a Random Forest model with 100 estimators
- Evaluates performance using:
  - Confusion matrix
  - Classification report
- Accepts real-time user input to predict churn and show probability

---

## 📦 Requirements

- Python 3.8+
- pandas
- scikit-learn

Install dependencies:
```bash
pip install pandas scikit-learn
```
## 📁 Dataset
File: ```Churn.csv```

### Contains:

- Demographics (gender, senior citizen)

- Account info (tenure, contract type, payment method)

- Service usage (streaming, online security, etc.)

- Target: Churn

## ▶️ How to Run
1. Place ```Churn.csv``` in the same folder as the script.

2. Run the script:
   ```
   python main.py
   ```
3. Enter customer details when prompted:
   ```
   Enter gender (Male/Female): Female
   Is the customer a senior citizen? (0 for No, 1 for Yes): 0
   ...
   Enter monthly charges: 70.35
   Enter total charges: 1395.45
   Prediction: Not Churn
   Probability of Churning: 0.23
   ```
## ✅ Example Output
   ```
   Confusion Matrix:
[[970  80]
 [140 219]]

Classification Report:
              precision    recall  f1-score   support

           0       0.87      0.92      0.89      1050
           1       0.73      0.61      0.67       359

    accuracy                           0.84      1409
   macro avg       0.80      0.76      0.78      1409
weighted avg       0.83      0.84      0.83      1409
```

## 🧠 How It Works
- Reads and preprocesses data:

- Converts ```TotalCharges``` to numeric

- Applies label encoding to categorical features

- Trains a Random Forest on train/test split (80/20)

- Predicts churn on test data

- Accepts new user input, encodes it, and predicts churn with probability

## ✏️ Author

Built and documented by [nandhini1424](https://github.com/nandhini1424)

