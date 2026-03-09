import streamlit as st
import joblib
import numpy as np

# Load trained models
bat_model = joblib.load("batting_model.pkl")
bowl_model = joblib.load("bowling_model.pkl")

st.title("🏏 Cricket Player Performance Predictor")

# choose prediction type
choice = st.radio("Select Prediction Type", ["Batting", "Bowling"])


# BATTING

if choice == "Batting":

    st.header("Predict Runs")

    mat = st.number_input("Matches Played", value=None, placeholder="Enter matches")
    inns = st.number_input("Innings Played", value=None, placeholder="Enter innings")
    sr = st.number_input("Strike Rate", value=None, placeholder="Enter strike rate")
    hundreds = st.number_input("Hundreds", value=None, placeholder="Enter hundreds")
    fifties = st.number_input("Fifties", value=None, placeholder="Enter fifties")
    ducks = st.number_input("Ducks", value=None, placeholder="Enter ducks")

    if st.button("Predict Runs"):

        features = np.array([[mat, inns, sr, hundreds, fifties, ducks]])

        prediction = bat_model.predict(features)

        st.success(f"Predicted Runs: {round(prediction[0])}")


# BOWLING 

if choice == "Bowling":

    st.header("Predict Wickets")

    mat = st.number_input("Matches Played", value=None, placeholder="Enter matches")
    inns = st.number_input("Innings Bowled", value=None, placeholder="Enter innings")
    balls = st.number_input("Balls Bowled", value=None, placeholder="Enter balls")
    mdns = st.number_input("Maidens", value=None, placeholder="Enter maidens")
    runs = st.number_input("Runs Conceded", value=None, placeholder="Enter runs")
    econ = st.number_input("Economy Rate", value=None, placeholder="Enter economy")
    five = st.number_input("5 Wicket Hauls", value=None, placeholder="Enter 5W")

    if st.button("Predict Wickets"):

        features = np.array([[mat, inns, balls, mdns, runs, econ, five]])

        prediction = bowl_model.predict(features)

        st.success(f"Predicted Wickets: {round(prediction[0])}")