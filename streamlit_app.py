import streamlit as st
import joblib
import numpy as np

# Load models
clf = joblib.load("classifier.pkl")
reg = joblib.load("regressor.pkl")

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")
st.title("🎓 Student Performance Predictor")

math = st.slider("Math Score", 0, 100, 50)
reading = st.slider("Reading Score", 0, 100, 50)
writing = st.slider("Writing Score", 0, 100, 50)

if st.button("Predict"):
    input_data = np.array([[math, reading, writing]])
    prediction = clf.predict(input_data)[0]
    estimated_score = reg.predict(input_data)[0]
    
    result = "✅ Pass" if prediction == 1 else "❌ Fail"
    st.subheader(f"Prediction: {result}")
    st.write(f"📊 Estimated Score: **{estimated_score:.2f}**")
