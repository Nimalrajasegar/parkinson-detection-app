import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("Parkinson's Detection App")

st.write("Enter values to predict")

# Example inputs (you can change later)
f1 = st.number_input("Feature 1")
f2 = st.number_input("Feature 2")

if st.button("Predict"):
    result = model.predict([[f1, f2]])

    if result[0] == 1:
        st.error("Parkinson's Detected")
    else:
        st.success("Healthy")
