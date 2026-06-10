import streamlit as st
import joblib
import numpy as np

# # Load model
# with open('model.pkl', 'rb') as file:
#     model = pickle.load(file)

model = joblib.load("model.pkl")

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺")

st.title("🩺 Diabetes Prediction App")
st.write("Enter patient details to predict whether the person is diabetic.")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
glucose = st.number_input("Glucose", min_value=0.0)
blood_pressure = st.number_input("Blood Pressure", min_value=0.0)
skin_thickness = st.number_input("Skin Thickness", min_value=0.0)
insulin = st.number_input("Insulin", min_value=0.0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
age = st.number_input("Age", min_value=1, step=1)

if st.button("Predict"):

    features = np.array([
        [
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            dpf,
            age
        ]
    ])

    prediction = model.predict(features)[0]

    try:
        probability = model.predict_proba(features)[0][1]
    except:
        probability = None

    if prediction == 1:
        st.error("⚠️ Prediction: Diabetic")
    else:
        st.success("✅ Prediction: Not Diabetic")

    if probability is not None:
        st.write(f"**Probability of Diabetes:** {probability:.2%}")