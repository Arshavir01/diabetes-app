import pickle
from pathlib import Path
 
import numpy as np
import streamlit as st
 
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
    st.warning('Educational demo only. Not medical advice.')
 
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')
 
    if st.button('Diabetes Test Result'):
        try:
            diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness,
                                             Insulin, BMI, DiabetesPedigreeFunction, Age])
            st.success(diagnosis)
        except ValueError:
            st.error('Please fill in all fields with numbers.')
 
 
if __name__ == '__main__':
    main()
