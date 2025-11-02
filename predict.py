import joblib
import numpy as np

def Predict(input_data):
    # Load the saved model and scaler
    knn_bc_model = joblib.load('breast_cancer_knn_model.joblib')
    scalar_bc = joblib.load('breast_cancer_scaler.joblib')

    # Scale the input data
    input_data_scaled = scalar_bc.transform(input_data)

    # Predict (returns 0 or 1)
    prediction = knn_bc_model.predict(input_data_scaled)

    # Convert numeric prediction to label 'B' or 'M'
    predicted_label = 'B' if prediction[0] == 0 else 'M'

    return predicted_label
