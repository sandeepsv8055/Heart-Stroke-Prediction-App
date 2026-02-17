import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
.big-title {
    font-size: 42px;
    font-weight: bold;
    color: #e63946;
}
.subtitle {
    font-size: 18px;
    margin-bottom: 30px;
}
.card {
    padding: 20px;
    border-radius: 12px;
    background-color: white;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="big-title">❤️ Heart Disease Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Machine Learning powered risk assessment using K-Nearest Neighbors</div>', unsafe_allow_html=True)

st.markdown("---")

# ---------------- SIDEBAR INPUT ----------------
st.sidebar.header("📝 Enter Patient Details")

age = st.sidebar.slider("Age", 18, 100, 40)
sex = st.sidebar.selectbox("Sex", ["M", "F"])
chest_pain = st.sidebar.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.sidebar.number_input("Resting Blood Pressure", 80, 200, 120)
cholesterol = st.sidebar.number_input("Cholesterol", 100, 600, 200)
fasting_bs = st.sidebar.selectbox("Fasting Blood Sugar > 120", [0, 1])
resting_ecg = st.sidebar.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.sidebar.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.sidebar.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.sidebar.slider("Oldpeak", 0.0, 6.0, 1.0)
st_slope = st.sidebar.selectbox("ST Slope", ["Up", "Flat", "Down"])

predict_button = st.sidebar.button("🔍 Predict Risk")

# ---------------- MAIN DASHBOARD ----------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 Patient Overview")
    st.metric("Age", age)
    st.metric("Resting BP", resting_bp)
    st.metric("Cholesterol", cholesterol)

with col2:
    st.markdown("### ⚙️ Model Information")
    st.metric("Model Used", "KNN")
    st.metric("Deployment", "Streamlit")
    st.metric("Status", "Active")

# ---------------- PREDICTION ----------------
if predict_button:

    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    st.markdown("---")
    st.markdown("## 🩺 Prediction Result")

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
        st.progress(float(probability))
        st.write(f"### Risk Probability: {probability*100:.2f}%")
    else:
        st.success("✅ Low Risk of Heart Disease")
        st.progress(float(probability))
        st.write(f"### Risk Probability: {probability*100:.2f}%")

# ---------------- ABOUT SECTION ----------------
st.markdown("---")
with st.expander("ℹ️ About This Project"):
    st.write("""
    This application compares multiple supervised learning algorithms 
    and selects the best-performing model (KNN) for heart disease prediction.
    
    Built using:
    - Python
    - Scikit-learn
    - Streamlit
    
    ⚠️ This tool is for educational purposes only and not a medical diagnosis.
    """)
