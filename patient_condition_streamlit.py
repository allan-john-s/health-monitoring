
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import os

# Load dataset and train model
data = pd.read_csv("/Users/allanjohn/datawarehiusinh qa1/Health-Monitoring-system-by-using-Machine-Learning/iot_dataset.csv")
X = data[["Patient ID", "Temperature Data", "ECG Data", "Pressure Data"]]
Y = data["Target"]
DT_Algorithm = DecisionTreeClassifier()
DT_Algorithm.fit(X, Y)

st.title("Patient Condition Prediction")

# Input fields for user data
patient_id = st.number_input("Patient ID", min_value=0)
temperature = st.number_input("Temperature Data", min_value=0)
ecg = st.number_input("ECG Data", min_value=0)
pressure = st.number_input("Pressure Data", min_value=0)
# Remove 'Other Feature' since not in training data

if st.button("Predict Condition"):
    input_array = np.array([patient_id, temperature, ecg, pressure]).reshape(1, -1)
    prediction = DT_Algorithm.predict(input_array)
    if prediction[0] == 0:
        st.success("The Patient Condition is Low")
    elif prediction[0] == 1:
        st.success("The Patient Condition is Medium")
    else:
        st.success("The Patient Condition is High")
