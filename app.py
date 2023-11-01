import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('movies.pkl')

# Define a function to predict popularity
def predict_popularity(vote_average, vote_count):
    new_data = np.array([[vote_average, vote_count]])
    scaled_new_data = scaler.transform(new_data)
    predicted_popularity = model.predict(scaled_new_data)
    return predicted_popularity[0][0]

# Streamlit UI
st.title('Movie Popularity Prediction')

# Input fields for user to enter vote_average and vote_count
vote_average = st.slider('Vote Average', 0.0, 10.0, step=0.1)
vote_count = st.number_input('Vote Count', min_value=0, max_value=10000, step=1)

# Predict button
if st.button('Predict Popularity'):
    popularity_prediction = predict_popularity(vote_average, vote_count)
    st.write(f'Predicted Popularity: {popularity_prediction:.2f}')
