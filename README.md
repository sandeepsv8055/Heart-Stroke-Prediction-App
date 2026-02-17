# ❤️ Heart Stroke Prediction App

A Machine Learning powered web application that predicts the risk of heart disease based on medical parameters.

Built using **Python, Scikit-learn, and Streamlit** and deployed as an interactive web app.

---

## 🚀 Project Overview

This project compares multiple supervised machine learning algorithms to predict heart disease risk.

After evaluating different models, **K-Nearest Neighbors (KNN)** achieved the best performance and was selected for deployment.

The final model is integrated into a Streamlit web application for real-time predictions.

---

## 📊 Machine Learning Workflow

1. Data Cleaning & Preprocessing
2. Exploratory Data Analysis (EDA)
3. Feature Encoding & Scaling
4. Model Training & Comparison
5. Performance Evaluation
6. Model Deployment

---

## 🤖 Models Compared

The following supervised learning algorithms were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

### ✅ Best Performing Model: **KNN**

KNN achieved the highest accuracy and better generalization compared to other models.

---

## 🧠 Features Used

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise-Induced Angina
- ST Depression (Oldpeak)
- ST Slope

---

## 🛠 Tech Stack

- Python
- Scikit-learn
- Pandas
- NumPy
- Joblib
- Streamlit

---

## 📂 Project Structure
HeartDiseaseDetection/ │ ├── app.py ├── requirements.txt ├── KNN_heart.pkl ├── scaler.pkl ├── columns.pkl └── HeartDiseasePrediction.ipynb

## ▶️ How to Run Locally


git clone <your-repo-link>
cd HeartDiseaseDetection
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py

## 🌐 Deployment

The application is deployed using Streamlit Community Cloud.

## ⚠️ Disclaimer

This application is for educational purposes only and should not be used for medical diagnosis.


## KNN achieved an accuracy of 89% on the test dataset.
