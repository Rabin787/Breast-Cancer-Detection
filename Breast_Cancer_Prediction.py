import streamlit as st
import numpy as np

# App title and description
st.title("🩷 Breast Cancer Detection System")
st.write("Enter the following cell nucleus measurements to predict whether the tumor is **Benign (B)** or **Malignant (M)**.")

# Input fields for all 30 features
radius_mean = st.number_input("Radius Mean", value=14.0)
texture_mean = st.number_input("Texture Mean", value=19.0)
perimeter_mean = st.number_input("Perimeter Mean", value=90.0)
area_mean = st.number_input("Area Mean", value=600.0)
smoothness_mean = st.number_input("Smoothness Mean", value=0.1)
compactness_mean = st.number_input("Compactness Mean", value=0.12)
concavity_mean = st.number_input("Concavity Mean", value=0.08)
concave_points_mean = st.number_input("Concave Points Mean", value=0.05)
symmetry_mean = st.number_input("Symmetry Mean", value=0.18)
fractal_dimension_mean = st.number_input("Fractal Dimension Mean", value=0.06)
radius_se = st.number_input("Radius SE", value=0.4)
texture_se = st.number_input("Texture SE", value=1.2)
perimeter_se = st.number_input("Perimeter SE", value=2.0)
area_se = st.number_input("Area SE", value=40.0)
smoothness_se = st.number_input("Smoothness SE", value=0.007)
compactness_se = st.number_input("Compactness SE", value=0.025)
concavity_se = st.number_input("Concavity SE", value=0.03)
concave_points_se = st.number_input("Concave Points SE", value=0.005)
symmetry_se = st.number_input("Symmetry SE", value=0.02)
fractal_dimension_se = st.number_input("Fractal Dimension SE", value=0.003)
radius_worst = st.number_input("Radius Worst", value=16.0)
texture_worst = st.number_input("Texture Worst", value=25.0)
perimeter_worst = st.number_input("Perimeter Worst", value=105.0)
area_worst = st.number_input("Area Worst", value=800.0)
smoothness_worst = st.number_input("Smoothness Worst", value=0.13)
compactness_worst = st.number_input("Compactness Worst", value=0.25)
concavity_worst = st.number_input("Concavity Worst", value=0.27)
concave_points_worst = st.number_input("Concave Points Worst", value=0.12)
symmetry_worst = st.number_input("Symmetry Worst", value=0.3)
fractal_dimension_worst = st.number_input("Fractal Dimension Worst", value=0.08)

# Prepare input data
input_data = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean,
                        smoothness_mean, compactness_mean, concavity_mean,
                        concave_points_mean, symmetry_mean, fractal_dimension_mean,
                        radius_se, texture_se, perimeter_se, area_se,
                        smoothness_se, compactness_se, concavity_se,
                        concave_points_se, symmetry_se, fractal_dimension_se,
                        radius_worst, texture_worst, perimeter_worst, area_worst,
                        smoothness_worst, compactness_worst, concavity_worst,
                        concave_points_worst, symmetry_worst, fractal_dimension_worst]])

btn=st.button("Predict Breast Cancer")
if btn:
    from predict import Predict
    result = Predict(input_data)
    label_map = {'M': "Malignant : Cancerous tumor. Grows and spreads to other parts of the body. Requires urgent medical treatment. ",
                 'B': "Benign : Non-cancerous tumor. Does not spread. Usually less dangerous and often removable."}
    prediction = label_map[result[0]]
    st.info(prediction)