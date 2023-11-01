import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('movies.pkl')

# Streamlit UI
st.title("Movie Popularity Prediction")

# Input for vote_average and vote_count
vote_average = st.number_input("Vote Average:", min_value=0.0, max_value=10.0, step=0.1)
vote_count = st.number_input("Vote Count:", min_value=0, step=1)

# Predict button
if st.button("Predict Popularity"):
    # Scale the input data
    new_data = np.array([[vote_average, vote_count]])
    predicted_popularity = model.predict(new_data)

    st.write("Predicted Popularity:", predicted_popularity[0][0]/10000)