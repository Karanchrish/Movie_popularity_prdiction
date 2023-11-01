import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

# Load the model
model = keras.models.load_model('movies.h5')

# Load the scaler
scaler = StandardScaler()
scaler.mean_ = np.array([7.0, 500])
scaler.scale_ = np.array([1.0, 1.0])

st.title('Movie Popularity Prediction App')

# Input for user to enter vote_average and vote_count
vote_average = st.slider('Vote Average:', min_value=0.0, max_value=10.0, step=0.1, value=7.0)
vote_count = st.slider('Vote Count:', min_value=0, max_value=10000, step=10, value=500)

# Transform the input data
new_data = np.array([[vote_average, vote_count]])
scaled_new_data = scaler.transform(new_data)

# Predict the popularity
predicted_popularity = model.predict(scaled_new_data)[0][0]

# Display the prediction
st.write("Predicted Popularity:", predicted_popularity)
