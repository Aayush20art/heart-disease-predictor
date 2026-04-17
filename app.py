import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide"
)

st.markdown("""
<style>

.main {
    background-color:#f8fafc;
}

h1 {
    background: linear-gradient(90deg,#2563eb,#06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align:center;
}

.result-card {
    background-color:white;
    padding:25px;
    border-radius:15px;
    text-align:center;
    box-shadow:0px 4px 15px rgba(0,0,0,0.1);
}

.stButton>button {
    background: linear-gradient(90deg,#2563eb,#06b6d4);
    color:white;
    border-radius:10px;
    height:45px;
    width:100%;
    font-size:16px;
}

</style>
""", unsafe_allow_html=True)

model = joblib.load('Logistic_heart.pkl')
scaler = joblib.load('scaler_heart.pkl')
expected_columns = joblib.load('columns.pkl')

st.markdown("""
<h1>🫀 Heart Disease Prediction System</h1>
<p style='text-align:center;color:#475569;'>AI powered system to estimate heart disease risk</p>
""", unsafe_allow_html=True)

st.divider()

st.sidebar.header("🩺 Patient Information")

age = st.sidebar.slider("Age", 18, 100, 40)
sex = st.sidebar.selectbox("Sex", ['Male', 'Female'])
Chest_pain = st.sidebar.selectbox("Chest Pain Type", ['ATA', 'NAP', 'TA', 'ASY'])
Resting_bp = st.sidebar.number_input("Resting Blood Pressure", 80, 200, 120)
Cholesterol = st.sidebar.number_input("Cholesterol", 100, 600, 200)
fasting_bs = st.sidebar.selectbox("Fasting Blood Sugar >120", [0,1])
Resting_ECG = st.sidebar.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.sidebar.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.sidebar.selectbox("Exercise Angina", ['Y', 'N'])
oldpeak = st.sidebar.slider("Oldpeak", 0.0, 6.0, 1.0)
st_slope = st.sidebar.selectbox("ST Slope", ["Up", "Flat", "Down"])

st.subheader("📊 Prediction Result")

if st.button("🔍 Predict Heart Disease Risk"):

    raw_input = {
        "Age": age,
        "RestingBP": Resting_bp,
        "Cholesterol": Cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_" + sex: 1,
        "ChestPainType_" + Chest_pain: 1,
        "RestingECG_" + Resting_ECG: 1,
        "ExerciseAngina_" + exercise_angina: 1,
        "ST_Slope_" + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]

    st.divider()

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
        st.progress(90)
        st.markdown("### 🚨 Please consult a cardiologist.")
    else:
        st.success("✅ Low Risk of Heart Disease")
        st.progress(30)
        st.markdown("### ❤️ Heart condition looks normal.")

st.divider()
st.caption("Developed by Aayush Sharma | Data Science")
