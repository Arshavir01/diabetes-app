#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 17:52:31 2026

@author: arshavir
"""

import numpy as np
import pickle
import streamlit as st
from pathlib import Path

BASE = Path(__file__).parent

loaded_model = pickle.load(open(BASE / 'diabetes_trained_model.sav', 'rb'))
scaler = pickle.load(open(BASE / 'scaler.sav', 'rb'))

def diabetes_prediction(input_data):
    arr = np.asarray(input_data, dtype=float).reshape(1, -1)   # strings -> floats
    arr = scaler.transform(arr)                                # same scaling as training
    prediction = loaded_model.predict(arr)

    if prediction[0] == 0:
        return 'The person is NOT diabetic'
    return 'The person is diabetic'
    
    
def main(): 
    
    st.title('Diabetes Prediction Web app')

    
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')
    
    
    diagnosis = ''
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
    st.success(diagnosis)
    
    
if __name__ == '__main__':
    main()