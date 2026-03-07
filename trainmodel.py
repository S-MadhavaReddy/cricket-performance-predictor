import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load Excel file
file = "cricketfappp.xlsx"

batting = pd.read_excel(file, sheet_name="batting")
bowling = pd.read_excel(file, sheet_name="bowling")

# BATTING MODEL

X_batting = batting[["Mat", "Inns", "SR", "Hundreds", "Fifties", "Ducks"]]

y_batting = batting["Runs"]

bat_model = RandomForestRegressor()

bat_model.fit(X_batting, y_batting)

joblib.dump(bat_model, "batting_model.pkl")

print("Batting model trained")


# BOWLING MODEL

X_bowling = bowling[["Mat", "Inns", "Balls", "Mdns", "Runs", "Econ", "5w"]]

y_bowling = bowling["Wkts"]

bowl_model = RandomForestRegressor()

bowl_model.fit(X_bowling, y_bowling)

joblib.dump(bowl_model, "bowling_model.pkl")

print("Bowling model trained")