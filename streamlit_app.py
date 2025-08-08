import streamlit as st
import tensorflow as tf
import numpy as np

# Load pre-trained model (cached for performance)
@st.cache_resource
def load_model():
    # CRITICAL FIX: Add compile=False to avoid metrics deserialization error
    return tf.keras.models.load_model('model.h5', compile=False)

model = load_model()

# 1. Hero Section
st.title("💰 Medical Insurance Cost Predictor")
st.caption("Predict your annual healthcare expenses | TensorFlow Neural Network")

# 2. Input Panel (Sidebar)
with st.sidebar:
    st.header("Your Profile")
    age = st.slider("Age", 18, 65, 35)
    sex = st.radio("Sex", ["Female", "Male"])
    bmi = st.number_input("BMI", 15.0, 50.0, 25.0, 0.1)
    children = st.slider("Children", 0, 5, 0)
    smoker = st.radio("Smoker", ["No", "Yes"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])
    
    # Convert inputs to model format
    sex = 0 if sex == "Female" else 1
    smoker = 0 if smoker == "No" else 1
    region_map = {"northeast":1, "northwest":0, "southeast":2, "southwest":3}
    region = region_map[region]

# 3. Prediction Output
if st.button("Calculate Prediction"):
    input_data = np.array([[age, sex, bmi, children, smoker, region]])
    prediction = model.predict(input_data)[0][0]
    
    st.subheader("Predicted Annual Cost")
    st.metric(label="", value=f"${prediction:,.2f}")
    
    st.caption(f"Model accuracy: ±${3069:,.0f} (MAE)")
