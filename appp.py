import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load models
bat_model = joblib.load("batting_model.pkl")
bowl_model = joblib.load("bowling_model.pkl")

# Load Excel sheets
bat_df = pd.read_excel("cricketfappp.xlsx", sheet_name="batting")
bowl_df = pd.read_excel("cricketfappp.xlsx", sheet_name="bowling")

st.title("🏏 Cricket Player Performance Predictor")

choice = st.radio("Select Prediction Type", ["Batting", "Bowling"])


# ---------------- BATTING ---------------- #

if choice == "Batting":

    st.header("Batting Prediction")

    players = ["Select Player"] + list(bat_df["Player"].unique())
    player = st.selectbox("Select Player", players)

    if player == "Select Player":

        st.write("Player Stats Used For Prediction")

        empty_data = {"Mat":[0], "Inns":[0], "SR":[0], "Hundreds":[0], "Fifties":[0], "Ducks":[0]}

        st.dataframe(empty_data)

    else:

        player_data = bat_df[bat_df["Player"] == player]

        mat = player_data["Mat"].values[0]
        inns = player_data["Inns"].values[0]
        sr = player_data["SR"].values[0]
        hundreds = player_data["Hundreds"].values[0]
        fifties = player_data["Fifties"].values[0]
        ducks = player_data["Ducks"].values[0]

        st.write("Player Stats Used For Prediction")

        st.dataframe(player_data[["Mat","Inns","SR","Hundreds","Fifties","Ducks"]])

        if st.button("Predict Runs"):

            features = np.array([[mat, inns, sr, hundreds, fifties, ducks]])

            prediction = bat_model.predict(features)

            st.success(f"Predicted Runs: {round(prediction[0])}")


# ---------------- BOWLING ---------------- #

if choice == "Bowling":

    st.header("Bowling Prediction")

    players = ["Select Player"] + list(bowl_df["Player"].unique())
    player = st.selectbox("Select Player", players)

    if player == "Select Player":

        st.write("Player Stats Used For Prediction")

        empty_data = {"Mat":[0], "Inns":[0], "Balls":[0], "Mdns":[0], "Runs":[0], "Econ":[0], "5w":[0]}

        st.dataframe(empty_data)

    else:

        player_data = bowl_df[bowl_df["Player"] == player]

        mat = player_data["Mat"].values[0]
        inns = player_data["Inns"].values[0]
        balls = player_data["Balls"].values[0]
        mdns = player_data["Mdns"].values[0]
        runs = player_data["Runs"].values[0]
        econ = player_data["Econ"].values[0]
        five = player_data["5w"].values[0]

        st.write("Player Stats Used For Prediction")

        st.dataframe(player_data[["Mat","Inns","Balls","Mdns","Runs","Econ","5w"]])

        if st.button("Predict Wickets"):

            features = np.array([[mat, inns, balls, mdns, runs, econ, five]])

            prediction = bowl_model.predict(features)

            st.success(f"Predicted Wickets: {round(prediction[0])}")


st.markdown("---")
st.markdown("Built using Machine Learning + Streamlit")