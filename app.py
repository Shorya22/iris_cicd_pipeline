import streamlit as st
import joblib
import numpy as np
from sklearn.datasets import load_iris

# Load model
model = joblib.load("model.pkl")
iris = load_iris()

st.title("🌸 Iris Classifier - Streamlit + Docker + CI/CD")

st.write("Enter measurements to classify iris species:")

# Input sliders
sl = st.slider("Sepal Length",min_value= 0.0, max_value= 10.0 )
sw = st.slider("Sepal Width", min_value= 0.0, max_value= 10.0 )
pl = st.slider("Petal Length",min_value= 0.0, max_value= 10.0 )
pw = st.slider("Petal Width", min_value= 0.0, max_value= 10.0 )    

if st.button("Predict"):
    features = np.array([[sl, sw, pl, pw]])
    pred = model.predict(features)[0]
    st.success(f"🌿 Predicted Species: {iris.target_names[pred]}")