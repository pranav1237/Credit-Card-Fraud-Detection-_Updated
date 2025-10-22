# 💳 Credit Card Fraud Detection

A machine learning project designed to detect fraudulent credit card transactions using data analysis, feature engineering, and classification models.  
The goal is to help identify potentially fraudulent activity with high precision and minimal false positives.

---

## 🚀 Project Overview

Fraudulent transactions are rare but costly.  
This project applies machine learning techniques on a highly imbalanced dataset (like the [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)) to build a predictive model that can classify transactions as **fraudulent** or **legitimate**.

---

## 🧠 Key Features

- **Data Preprocessing:** Cleans and normalizes input data, handles missing values, and manages class imbalance using undersampling/oversampling.
- **Exploratory Data Analysis (EDA):** Visualizes data distributions and correlations.
- **Model Training:** Uses Logistic Regression, Random Forest, and XGBoost for comparison.
- **Evaluation:** Reports accuracy, precision, recall, F1-score, and ROC-AUC.
- **Deployment (Optional):** Flask or FastAPI web app for real-time fraud prediction.

- Create and activate a virtual environment
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate 

# Install dependencies
pip install -r requirements.txt
pip install -r requirements_web.txt

# Run the Flask API
python src/app.py

## You Can Acess the full Project on the given Link:  
## Link:  https://drive.google.com/file/d/1OD8-lV5_0EBiFR-AUnVSh5929E1vaiWi/view?usp=drive_link
