# import libraries

import streamlit as st
import pandas as pd
import pickle

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

clf_reduced = pickle.load(open("EPL_model_reduced_Predictor.p", "rb"))
data_for_avg = pickle.load(open("data_for_avg.p", "rb"))

# Add title and instructions
st.title("Premier League Match Prediction")
st.text("By Wylie")

st.subheader("Select the Home and Away teams, along with their team weights to boost the probability. Click submit and scroll down likelihood to of match result")

team1_name = st.selectbox(
    "Home Team Name:",
    ("Arsenal", "Aston Villa", 
     "Bournemouth", "Brentford","Burnley", "Brighton",
     "Chelsea", "Crystal Palace",
     "Everton",
     "Fulham",
     "Leeds","Liverpool",
     "Manchester City", "Manchester Utd",
     "Newcastle", "Nott'm Forest", 
     "Spurs", 
     "Wolves"
     ), key="team1_name"
)

team1_weight = st.number_input(
    label = "Home team weight",
    min_value = 0.0,
    max_value = 2.0,
    value = 1.0,
    key = "team1_weight")

team2_name = st.selectbox(
    "Away Team Name:",
    ("Arsenal", "Aston Villa", 
     "Bournemouth", "Brentford","Burnley", "Brighton",
     "Chelsea", "Crystal Palace",
     "Everton",
     "Fulham",
     "Leeds","Liverpool",
     "Manchester City", "Manchester Utd",
     "Newcastle", "Nott'm Forest", 
     "Spurs", 
     "Wolves"
     ),  key="team2_name"
)

team2_weight = st.number_input(
    label = "Away team weight",
    min_value = 0.0,
    max_value = 2.0,
    value = 1.0,
    key = "team2_weight")

# submit inputs to model


if st.button("Submit for Prediction"):
    
    # store the input data into a dataframe for prediction
 
    # First collect all data for each team in to separate dataframes.
    tmp_team1 = data_for_avg[data_for_avg['Team'] == team1_name].copy()
    tmp_team1.drop(['Team'], axis=1, inplace = True)

    tmp_team2 = data_for_avg[data_for_avg['Team'] == team2_name].copy()
    tmp_team2.drop(['Team'], axis=1, inplace = True)

    # Average numerical data in each datafram and put into a new dataframe.    
    averaged_data = tmp_team1.mean().to_frame().T
    tmp_team1_mean = averaged_data.reset_index(drop=True)

    averaged_data = tmp_team2.mean().to_frame().T
    tmp_team2_mean = averaged_data.reset_index(drop=True)

    # Combine average data dataframes and reset the index
    combined_avg = pd.concat([tmp_team1_mean, tmp_team2_mean], axis=0).reset_index(drop=True)

    # Ensure the order of features matches the training data
    combined_avg =combined_avg[tmp_team1.columns]
    
    
    # apply model pipeline to the input data and make a prediction
    
    predict_proba = clf_reduced.predict_proba(combined_avg)
    
    # Extract probabilities
    home_lose, home_tie, home_win = predict_proba[0]
    away_lose, away_tie, away_win = predict_proba[1]
    
    # Calculate overall probabilities
    home_win_prob = ((home_win + away_lose)/2) * team1_weight
    away_win_prob = ((away_win + home_lose)/2) * team2_weight
    tie_prob = ((home_tie * team1_weight) + (away_tie * team2_weight))/2 # + home_win * away_win + home_lose * away_lose
    
    # Normalize probabilities due to a high precentage of conflicting outcomes
    total_prob = home_win_prob + away_win_prob + tie_prob
    home_win_prob /= total_prob
    away_win_prob /= total_prob
    tie_prob /= total_prob
    
    if home_win_prob > away_win_prob:
        pick = "Home Win"
    elif away_win_prob > home_win_prob:
        pick = "Away Win"
    else:
        pick = "Tie"
    
    home_win = round((home_win_prob * 100),2)
    tie = round((tie_prob * 100),2)
    away_win = round((away_win_prob * 100),2)


    #output prediction
    
    st.subheader("Our Prediction")
    st.subheader(f"Home Win: {home_win}")
    st.subheader(f"Tie: {tie}")
    st.subheader(f"Away Win: {away_win}") 
    st.subheader(f"Pick: {pick}")

    