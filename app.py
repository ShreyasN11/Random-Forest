import streamlit as st
import joblib
import numpy as np

model = joblib.load('random_forest_model.pkl')

st.title('IPL Score Prediction App')

wicket = st.number_input('Wickets', min_value=0,max_value=10)
overs = st.number_input('Overs', min_value=0, max_value=20)
runs = st.number_input('Runs', min_value=0)

if st.button('Predict'):
    input_data = np.array([[wicket, overs, runs]])
    prediction = model.predict(input_data)
    rounded_prediction = np.ceil(prediction)
    st.markdown(f"<div style='font-size:24px; padding:10px;'>Predicted Score: {rounded_prediction[0]}</div>", unsafe_allow_html=True)
