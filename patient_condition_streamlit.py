
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import os

# Page configuration
st.set_page_config(
    page_title="Health Monitoring System",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        height: 3em;
        border-radius: 10px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
    }
    .patient-header {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/000000/hospital-2.png", width=100)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Patient Monitoring"])

if page == "Patient Monitoring":
    # Header
    st.markdown("""
        <div class='patient-header'>
            <h1 style='text-align: center; color: #FF4B4B;'>🏥 Patient Health Monitoring System</h1>
            <p style='text-align: center; color: #666666;'>Advanced ML-powered health condition prediction system</p>
        </div>
    """, unsafe_allow_html=True)

    # Load dataset and train model
    try:
        data = pd.read_csv("iot_dataset.csv")
        X = data[["Patient ID", "Temperature Data", "ECG Data", "Pressure Data"]]
        Y = data["Target"]
        DT_Algorithm = DecisionTreeClassifier()
        DT_Algorithm.fit(X, Y)
    except FileNotFoundError:
        st.error("The file `iot_dataset.csv` was not found. Please ensure the dataset is in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading or training the model: {e}")
        st.stop()

    # Create three columns for better layout
    col1, col2, col3 = st.columns(3)

    # Input fields with better organization
    with col1:
        st.markdown("### 📋 Patient Information")
        patient_id = st.number_input("Patient ID", min_value=0, help="Enter the unique identifier for the patient")

    with col2:
        st.markdown("### 🌡️ Vital Signs")
        temperature = st.number_input("Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0, step=0.1,
                                    help="Normal body temperature is around 37°C")
        
    with col3:
        st.markdown("###  Cardiac Data")
        ecg = st.number_input("ECG Reading", min_value=0, help="Enter the ECG reading value")
        pressure = st.number_input("Blood Pressure", min_value=0, help="Enter the blood pressure reading")

    # Add some space
    st.markdown("<br>", unsafe_allow_html=True)

    # Prediction button with loading animation
    if st.button("📊 Analyze Patient Condition"):
        with st.spinner('Analyzing patient data...'):
            try:
                input_array = np.array([patient_id, temperature, ecg, pressure]).reshape(1, -1)
                prediction = DT_Algorithm.predict(input_array)
                
                # Create columns for the result display
                res_col1, res_col2 = st.columns([2, 1])
                
                with res_col1:
                    if prediction[0] == 0:
                        st.markdown("""
                            <div style='background-color: #90EE90; padding: 20px; border-radius: 10px;'>
                                <h3 style='color: #006400; margin: 0;'>🟢 Patient Condition: LOW RISK</h3>
                                <p style='color: #006400; margin: 0;'>The patient's vital signs are within normal ranges.</p>
                            </div>
                        """, unsafe_allow_html=True)
                    elif prediction[0] == 1:
                        st.markdown("""
                            <div style='background-color: #FFE5B4; padding: 20px; border-radius: 10px;'>
                                <h3 style='color: #FF8C00; margin: 0;'>🟡 Patient Condition: MODERATE RISK</h3>
                                <p style='color: #FF8C00; margin: 0;'>The patient requires regular monitoring.</p>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                            <div style='background-color: #FFB6C1; padding: 20px; border-radius: 10px;'>
                                <h3 style='color: #DC143C; margin: 0;'>🔴 Patient Condition: HIGH RISK</h3>
                                <p style='color: #DC143C; margin: 0;'>Immediate medical attention recommended!</p>
                            </div>
                        """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction failed: {e}")