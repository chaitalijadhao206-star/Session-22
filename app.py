import streamlit as st
import pandas as pd
import joblib

# Load Model
model = joblib.load("heart_model.pkl")
columns = joblib.load("columns.pkl")

st.title("Heart Disease Prediction")

# User Inputs
age = st.number_input("Age", 1, 120, 30)
sex = st.selectbox("Sex", [0, 1])
chest_pain = st.number_input("Chest Pain Type", 1, 4, 1)
resting_bp = st.number_input("Resting Blood Pressure", 50, 250, 120)
cholesterol = st.number_input("Serum Cholesterol", 100, 700, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar", [0, 1])
rest_ecg = st.number_input("Resting ECG", 0, 2, 0)
max_hr = st.number_input("Max Heart Rate", 50, 250, 150)
exercise_angina = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
st_segment = st.number_input("ST Segment", 1, 3, 1)
major_vessels = st.number_input("Major Vessels", 0, 3, 0)
thal = st.number_input("Thal", 3, 7, 3)

if st.button("Predict"):

    data = pd.DataFrame([[
        age,
        sex,
        chest_pain,
        resting_bp,
        cholesterol,
        fasting_bs,
        rest_ecg,
        max_hr,
        exercise_angina,
        oldpeak,
        st_segment,
        major_vessels,
        thal
    ]], columns=[
        "age",
        "sex",
        "chest pain type",
        "resting blood pressure",
        "serum cholestoral",
        "fasting blood sugar",
        "resting electrocardiographic results",
        "max heart rate",
        "exercise induced angina",
        "oldpeak",
        "ST segment",
        "major vessels",
        "thal"
    ])

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.success("Heart Disease: Yes")
    else:
        st.error("Heart Disease: No")