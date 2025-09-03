# import libraries

import streamlit as st
import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

clf_reduced = pickle.load(open("web_app_Predictor.p", "rb"))
data_for_avg = pickle.load(open("web_app_data.p", "rb"))


tab1, tab2 = st.tabs(["Make Prediction", "About"])

# Images to spruce things up

with tab1:
    st.image("images/crystal_palace_stadium.jpg")

    # Add title and instructions
    st.title("Premier League Match Prediction")
    st.text("Data Science and Machine Learning by Wylie")
    st.text("Data provided by FBref.com")
    st.text("All previous matches for model training are from the Premier League between 11/9/2024 and 8/25/2025.")
    
    st.subheader("Select the Home and Away teams, along with their team weights to boost the probability. Click submit and scroll down to see the likelihood and pick of the match result")
    st.text("Note: Team weights should range beteen 0 and 2. They are set by you based on your knowledge of activities since 8/25/2025 on the team. Examples might be, a change in strategy with recent success could get a weight of 2. However, injuries, transfers, player suspensions could get 0.5")
    
    left, right = st.columns([1,3], vertical_alignment="top")
    
    left.subheader("Select the home and way teams of the match.")
    team1_name = left.selectbox(
        "Home Team Name:",
        ("Arsenal", "Aston Villa", 
         "Bournemouth", "Brentford","Burnley", "Brighton",
         "Chelsea", "Crystal Palace",
         "Everton",
         "Fulham",
         "Leeds United","Liverpool",
         "Manchester City", "Manchester Utd",
         "Newcastle", "Nott'm Forest", 
         "Spurs", "Sunderland", 
         "West Ham", "Wolves"
         ), width=150, key="team1_name"
    )
    
    team1_weight = left.number_input(
        label = "Home team weight",
        min_value = 0.0,
        max_value = 2.0,
        value = 1.0,
        width=150, key = "team1_weight")
    
    
    
    team2_name = left.selectbox(
        "Away Team Name:",
        ("Arsenal", "Aston Villa", 
         "Bournemouth", "Brentford","Burnley", "Brighton",
         "Chelsea", "Crystal Palace",
         "Everton",
         "Fulham",
         "Leeds United","Liverpool",
         "Manchester City", "Manchester Utd",
         "Newcastle", "Nott'm Forest", 
         "Spurs", "Sunderland", 
         "West Ham", "Wolves"
         ), width=150, key="team2_name"
    )
    
    
    
    team2_weight = left.number_input(
        label = "Away team weight",
        min_value = 0.0,
        max_value = 2.0,
        value = 1.0,
        width=150, key = "team2_weight")
    
    
    
    # GW1 no weights, rely on historic data entirely.  Unsure how summer transfers will work out
    game_week1_baseline_list = [[1,"Liverpool", "Bournemouth", 1, 1],
                      [2,"Aston Villa", "Newcastle", 1, 1],
                      [3,"Brighton", "Fulham", 1, 1],
                      [4,"Sunderland", "West Ham", 1, 1],
                      [5,"Spurs", "Burnley", 1, 1],
                      [6,"Wolves", "Man City", 1, 1],
                      [7,"Nott'm Forest", "Brentford", 1, 1], 
                      [8,"Chelsea", "Crystal Palace", 1, 1],
                      [9,"Man Utd", "Arsenal", 1, 1],
                      [10,"Leeds United", "Everton", 1, 1]]
    game_week1_baseline = pd.DataFrame(game_week1_baseline_list, columns=["match ID","Home", "Away", "Home Weight", "Away Weight"])
    
    # GW2 Add weights now that we have seen everyone play once and better understand transfers
    game_week2_baseline_list = [[1,"West Ham", "Chelsea", 1, 1],
                      [2,"Man City", "Spurs", 1, 1],
                      [3,"Bournemouth", "Wolves", 1, 1],
                      [4,"Brentford", "Aston Villa", 1, 1],
                      [5,"Burnley", "Sunderland", 1, 1],
                      [6,"Arsenal", "Leeds United", 1, 1],
                      [7,"Crystal Palace", "Nott'm Forest", 1, 1], 
                      [8,"Everton", "Brighton", 1, 1],
                      [9,"Fulham", "Man Utd", 1, 1],
                      [10,"Newcastle", "Liverpool", 1, 1]]
    game_week2_baseline = pd.DataFrame(game_week2_baseline_list, columns=["match ID","Home", "Away", "Home Weight", "Away Weight"])
    
    game_week2_weighted_list = [[1,"West Ham", "Chelsea", 1, 1],
                      [2,"Man City", "Spurs", 1, 1],
                      [3,"Bournemouth", "Wolves", 1, 1],
                      [4,"Brentford", "Aston Villa", 1, 1],
                      [5,"Burnley", "Sunderland", 1, 1],
                      [6,"Arsenal", "Leeds United", 1, 1],
                      [7,"Crystal Palace", "Nott'm Forest", 1, 1], 
                      [8,"Everton", "Brighton", 2, 1],
                      [9,"Fulham", "Man Utd", 2, 1],
                      [10,"Newcastle", "Liverpool", 1, 1]]
    game_week2_weighted = pd.DataFrame(game_week2_weighted_list, columns=["match ID","Home", "Away", "Home Weight", "Away Weight"])
    
    # GW3 Continue with weights plus observer underestimates in better teams from last season
    game_week3_baseline_list = [[1,"Chelsea", "Fulham", 1, 1],
                      [2,"Man Utd", "Burnley", 1, 1],
                      [3,"Sunderland", "Brentford", 1, 1],
                      [4,"Spurs", "Bournemouth",1, 1],
                      [5,"Wolves", "Everton",1 , 1],
                      [6,"Leeds United", "Newcastle" ,1, 1],
                      [7,"Brighton", "Man City", 1, 1], 
                      [8,"Nott'm Forest", "West Ham", 1, 1],
                      [9,"Liverpool", "Arsenal", 1, 1],
                      [10,"Aston Villa", "Crystal Palace", 1, 1]]
    game_week3_baseline = pd.DataFrame(game_week3_baseline_list, columns=["match ID","Home", "Away", "Home Weight", "Away Weight"])
    
    game_week3_weighted_list = [[1,"Chelsea", "Fulham", 1, 1],
                      [2, "Man Utd", "Burnley", 1, 1],
                      [3, "Sunderland", "Brentford", 1, 1],
                      [4, "Spurs", "Bournemouth",1, 1],
                      [5, "Wolves", "Everton",1 , 1],
                      [6, "Leeds United", "Newcastle",1, 1],
                      [7, "Brighton", "Man City", 1, 1], 
                      [8, "Nott'm Forest", "West Ham", 1, 2],
                      [9, "Liverpool", "Arsenal", 1, 1],
                      [10,"Aston Villa", "Crystal Palace", 1, 2]]
    game_week3_weighted = pd.DataFrame(game_week3_weighted_list, columns=["match ID","Home", "Away", "Home Weight", "Away Weight"])
    
    # GW4 Continue with weights plus one more based on last 3 GW performance
    game_week4_baseline_list = [[1,"Arsenal", "Nott'm Forest", 1, 1],
                      [2,"Bournemouth", "Brighton", 1, 1],
                      [3,"Crystal Palace", "Sunderland", 1, 1],
                      [4,"Everton", "Aston Villa",1, 1],
                      [5,"Fulham", "Leeds United",1 , 1],
                      [6,"Newcastle", "Wolves" ,1, 1],
                      [7,"West Ham", "Spurs", 1, 1], 
                      [8,"Brentford", "Chelsea", 1, 1],
                      [9,"Burnley", "Liverpool", 1, 1],
                      [10,"Man City", "Man Utd", 1, 1]]
    game_week4_baseline = pd.DataFrame(game_week4_baseline_list, columns=["match ID","Home", "Away", "Home Weight", "Away Weight"])
    
    game_week4_weighted_list = [[1,"Arsenal", "Nott'm Forest", 2, 1],
                      [2,"Bournemouth", "Brighton", 2, 1],
                      [3,"Crystal Palace", "Sunderland", 1, 1],
                      [4,"Everton", "Aston Villa",2, 1],
                      [5,"Fulham", "Leeds United",1 , 1],
                      [6,"Newcastle", "Wolves" ,2, 1],
                      [7,"West Ham", "Spurs", 2, 1], 
                      [8,"Brentford", "Chelsea", 1, 1],
                      [9,"Burnley", "Liverpool", 1, 1],
                      [10,"Man City", "Man Utd", 1, 1]]
    game_week4_weighted = pd.DataFrame(game_week4_weighted_list, columns=["match ID","Home", "Away", "Home Weight", "Away Weight"])
    
    # Create a dictionary to map string names to actual DataFrames
    # match_week_mapping = {
    #     "game_week1_baseline": game_week1_baseline,
    #     "game_week2_baseline_list": game_week2_baseline,
    #     "game_week2_weighted_list": game_week2_weighted,
    #     "game_week3_baseline_list": game_week3_baseline,
    #     "game_week3_weighted_list": game_week3_weighted,
    #     "game_week4_baseline_list": game_week4_baseline,
    #     "game_week4_weighted_list": game_week4_weighted
    # }
    
    match_week_mapping = {
        "game_week1": game_week1_baseline,
        "game_week2": game_week2_weighted,
        "game_week3": game_week3_weighted,
        "game_week4": game_week4_weighted
    }
    
    
    # gw_num_match_lineup = right.selectbox(
    #     "Game week matches and my suggested weights:",
    #     ("game_week1_baseline",
    #      "game_week2_baseline_list",
    #      "game_week2_weighted_list",
    #      "game_week3_baseline_list",
    #      "game_week3_weighted_list",
    #      "game_week4_baseline_list",
    #      "game_week4_weighted_list"
    #      ),  key="gw_num_match_lineup")
    
    right.subheader("All game week matches and my suggested team weights.")
    
    gw_num_match_lineup = right.selectbox(
        "Pick a game week (note: match id for reference with actual results):",
        ("game_week1",
         "game_week2",
         "game_week3",
         "game_week4"
         ),  key="gw_num_match_lineup")
    
    # Get the selected DataFrame and display it
    selected_dataframe = match_week_mapping.get(gw_num_match_lineup)
    if selected_dataframe is not None:
        right.dataframe(selected_dataframe)
    else:
        right.write(f"DataFrame for {gw_num_match_lineup} not yet implemented")
    
    gw_1_actuals_list = [[1,"Home Win","Home Win"],
                    [2,"Tie", "Home Win"],
                    [3,"Tie", "Home Win"],
                    [4,"Home Win", "Home Win"],
                    [5,"Home Win", "Home Win"],
                    [6,"Away Win", "Away Win"],
                    [7,"Home Win", "Away Win"],
                    [8,"Tie", "Home Win"],
                    [9,"Away Win", "Away Win"],
                    [10,"Home Win", "Away Win"]]
    gw_1_actuals = pd.DataFrame(gw_1_actuals_list, columns=["match ID","Result","Predicted"])
    
    gw_2_actuals_list = [[1,"Away Win", "Away Win"],
                    [2,"Away Win", "Home Win"],
                    [3,"Home Win", "Home Win"],
                    [4,"Home Win", "Home Win", ],
                    [5,"Home Win", "Away Win"],
                    [6,"Home Win", "Home Win"],
                    [7,"Tie", "Home Win"],
                    [8,"Home Win", "Home Win"],
                    [9,"Tie", "Home Win"],
                    [10,"Away Win", "Away Win"]]
    gw_2_actuals = pd.DataFrame(gw_2_actuals_list, columns=["match ID","Result", "Predicted"])
    
    gw_3_actuals_list = [[1,"Home Win", "Home Win"],
                    [2,"Home Win", "Away Win"],
                    [3,"Home Win", "Home Win"],
                    [4,"Away Win", "Away Win"],
                    [5,"Away Win", "Away Win"],
                    [6,"Tie", "Tie"],
                    [7,"Home Win", "Home Win"],
                    [8,"Away Win", "Away Win"],
                    [9,"Home Win", "Home Win"],
                    [10,"Away Win", "Away Win"]]
    gw_3_actuals = pd.DataFrame(gw_3_actuals_list, columns=["match ID","Result","Predicted"])
    
    actuals_week_mapping = {
        "game_week1": gw_1_actuals,
        "game_week2": gw_2_actuals,
        "game_week3": gw_3_actuals,
    }
    
    right.subheader("Actual results from that game week. Predicted based on weights suggested in matches.")
    
    gw_num_actuls = right.selectbox(
        "Pick a game week (actual results listed by match id from the table above):",
        ("game_week1",
         "game_week2",
         "game_week3"
         ),  key="gw_num_actuls")
    
    # Get the selected DataFrame and display it
    selected_dataframe = actuals_week_mapping.get(gw_num_actuls)
    if selected_dataframe is not None:
        right.dataframe(selected_dataframe)
        num_right =len(selected_dataframe.loc[(selected_dataframe["Predicted"] == selected_dataframe["Result"])])
        total = len(selected_dataframe)
        accuracy = num_right/total
        right.text(f"Predicted correct accuracy: {accuracy:.0%}")
    else:
        right.write(f"DataFrame for {gw_num_match_lineup} not yet implemented")
        right.text("Predicted correct accuracy: NONE")
    
    # submit inputs to model
    
    
    if left.button("Submit for Prediction", icon=":material/online_prediction:"):
        
        # store the input data into a dataframe for prediction
        # For analysis only.
        # team1_name="Arsenal"
        # team2_name="Leeds United"
        # teams = data_for_avg['Team']
        
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
        # combined_avg.columns
        # clf_reduced.feature_names_in_
        # print("columns not in fit")
        # for col in combined_avg.columns:
        #     if col not in clf_reduced.feature_names_in_:
        #         print(col)
        # print("columns not in avg data")
        # for col in clf_reduced.feature_names_in_:
        #     if col not in combined_avg.columns:
        #         print(col)                
       
        
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
        
        left.subheader("Our Prediction")
        left.subheader(f"Home Win: {home_win}")
        left.subheader(f"Tie: {tie}")
        left.subheader(f"Away Win: {away_win}") 
        left.subheader(f"Pick: {pick}")
        
with tab2:
    st.image("images/crystal_palace_stadium.jpg")

    st.text("Describe the project")
