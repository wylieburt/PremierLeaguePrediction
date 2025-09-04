# import libraries

import streamlit as st
import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

clf_reduced = pickle.load(open("web_app_Predictor.p", "rb"))
data_for_avg = pickle.load(open("web_app_data.p", "rb"))

################################################
# Create tabs to display at top of each tab
################################################

tab1, tab2, tab3 = st.tabs(["Make Prediction", "Tables", "About Project"])

################################################
# Contents of tab1 - Make Prediction
################################################
with tab1:

    # Image at top of tab
    st.image("crystal_palace_stadium.jpg")

    # Add title and references
    st.title("Premier League Match Prediction")
    st.text("Data Science and Machine Learning by Wylie")
    st.text("Data provided by FBref.com")
    st.text("All previous matches for model training are from the Premier League between 11/9/2024 and 8/25/2025.")
    
    # Add high level instructions
    st.subheader("Select the Home and Away teams, along with their team weights to boost the probability. Click submit and scroll down to see the likelihood and pick of the match result")
    st.text("Note: Team weights should range beteen 0 and 2. They are set by you based on your knowledge of activities since 8/25/2025 on the team. Examples might be, a change in strategy with recent success could get a weight of 2. However, injuries, transfers, player suspensions could get 0.5")
    
    # Establish columns on tab to arrange content
    left, middle, right = st.columns([1,3,2], vertical_alignment="top")
    
    #############
    # Left column
    #############
    
    # Prediction input data
    left.subheader("Make a forecast")
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

    # Process prediction button
       
    if left.button("Process", icon=":material/online_prediction:"):
        
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
        
        # Random Forest prediction
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
        
        # Select the Pick
        if (home_win_prob > away_win_prob) and (home_win_prob > tie_prob):
            pick = "Home Win"
        elif (away_win_prob > home_win_prob) and (away_win_prob > tie_prob):
            pick = "Away Win"
        elif (tie_prob > home_win_prob) and (tie_prob > away_win_prob):
            pick = "Tie"
        elif home_win_prob == away_win_prob:
            pick = "Tie"
        
        # Convert to percentage and reduce number of decimals
        home_win = round((home_win_prob * 100),2)
        tie = round((tie_prob * 100),2)
        away_win = round((away_win_prob * 100),2)
    
    
        # Output prediction to tab.  Using markdown to set text color
        
        left.markdown(f"""
            <span style="color: red;">Our Prediction</span><br><br>
            <span style="color: red;">Home Win: {home_win}</span><br><br>
            <span style="color: red;">Tie: {tie}</span><br><br>
            <span style="color: red;">Away Win: {away_win}</span><br><br>
            <span style="color: red;">Pick: {pick}</span>
            """, unsafe_allow_html=True)    
    
    #############
    # Middle column
    #############
    
    # Create dataframes for each game week containing Match ID, Home Team, Away Team, Home Weight, Away Weight
    
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
    game_week1_baseline = pd.DataFrame(game_week1_baseline_list, columns=["M#","Home", "Away", "H-Wt", "A-Wt"])
    
    # GW2 Add weights now that we have seen everyone play once and better understand transfers   
    game_week2_weighted_list = [[1,"West Ham", "Chelsea", 1, 1],
                      [2,"Man City", "Spurs", 1, 1],
                      [3,"Bournemouth", "Wolves", 1, 1],
                      [4,"Brentford", "Aston Villa", 1, 1],
                      [5,"Burnley", "Sunderland", 1, 1],
                      [6,"Arsenal", "Leeds United", 2, 1],
                      [7,"Crystal Palace", "Nott'm Forest", 1, 1], 
                      [8,"Everton", "Brighton", 2, 1],
                      [9,"Fulham", "Man Utd", 1, 1],
                      [10,"Newcastle", "Liverpool", 1, 1]]
    game_week2_weighted = pd.DataFrame(game_week2_weighted_list, columns=["M#","Home", "Away", "H-Wt", "A-Wt"])
    
    # GW3 Continue with weights plus observer underestimates in better teams from last season   
    game_week3_weighted_list = [[1,"Chelsea", "Fulham", 1, 1],
                      [2, "Man Utd", "Burnley", 1, 1],
                      [3, "Sunderland", "Brentford", 1, 1],
                      [4, "Spurs", "Bournemouth",1, 1],
                      [5, "Wolves", "Everton",1 , 1],
                      [6, "Leeds United", "Newcastle",1, 1],
                      [7, "Brighton", "Man City", 1, 1], 
                      [8, "Nott'm Forest", "West Ham", 1, 2],
                      [9, "Liverpool", "Arsenal", 1, 1],
                      [10,"Aston Villa", "Crystal Palace", 1, 1]]
    game_week3_weighted = pd.DataFrame(game_week3_weighted_list, columns=["M#","Home", "Away", "H-Wt", "A-Wt"])
    
    # GW4 Continue with weights plus one more based on last 3 GW performance   
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
    game_week4_weighted = pd.DataFrame(game_week4_weighted_list, columns=["M#","Home", "Away", "H-Wt", "A-Wt"])
    
    # Create a dictionary to map string names to actual DataFrames
    match_week_mapping = {
        "game_week1": game_week1_baseline,
        "game_week2": game_week2_weighted,
        "game_week3": game_week3_weighted,
        "game_week4": game_week4_weighted
    }
    
    middle.subheader("Game Week matches with team weights.")
    
    # pull down list to select game week to display
    gw_num_match_lineup = middle.selectbox(
        "Pick a game week (note: match id for reference with actual results):",
        ("game_week1",
         "game_week2",
         "game_week3",
         "game_week4"
         ),  key="gw_num_match_lineup")
    
    # Get the selected DataFrame and display it
    selected_dataframe = match_week_mapping.get(gw_num_match_lineup)
    if selected_dataframe is not None:
        middle.dataframe(selected_dataframe, hide_index=True)
    else:
        middle.write(f"DataFrame for {gw_num_match_lineup} not yet implemented")

    # Create dataframes for each game week containing Match ID, Actual  Result, Predicted Result
   
    # Game week 1
    gw_1_actuals_list = [[1,"Home Win","Home Win"],
                    [2,"Tie", "Tie"],
                    [3,"Tie", "Home Win"],
                    [4,"Home Win", "Tie"],
                    [5,"Home Win", "Tie"],
                    [6,"Away Win", "Away Win"],
                    [7,"Home Win", "Away Win"],
                    [8,"Tie", "Home Win"],
                    [9,"Away Win", "Away Win"],
                    [10,"Home Win", "Tie"]]
    gw_1_actuals = pd.DataFrame(gw_1_actuals_list, columns=["match ID","Result","Predicted"])

    # Game week 2    
    gw_2_actuals_list = [[1,"Away Win", "Away Win"],
                    [2,"Away Win", "Home Win"],
                    [3,"Home Win", "Home Win"],
                    [4,"Home Win", "Home Win", ],
                    [5,"Home Win", "Tie"],
                    [6,"Home Win", "Home Win"],
                    [7,"Tie", "Tie"],
                    [8,"Home Win", "Home Win"],
                    [9,"Tie", "Tie"],
                    [10,"Away Win", "Away Win"]]
    gw_2_actuals = pd.DataFrame(gw_2_actuals_list, columns=["match ID","Result", "Predicted"])

    # game week 3
    gw_3_actuals_list = [[1,"Home Win", "Home Win"],
                    [2,"Home Win", "Tie"],
                    [3,"Home Win", "Away Win"],
                    [4,"Away Win", "Away Win"],
                    [5,"Away Win", "Tie"],
                    [6,"Tie", "Tie"],
                    [7,"Home Win", "Away Win"],
                    [8,"Away Win", "Away Win"],
                    [9,"Home Win", "Home Win"],
                    [10,"Away Win", "Away Win"]]
    gw_3_actuals = pd.DataFrame(gw_3_actuals_list, columns=["match ID","Result","Predicted"])
    
    # mapping of game selection text to the correct dataframe
    actuals_week_mapping = {
        "game_week1": gw_1_actuals,
        "game_week2": gw_2_actuals,
        "game_week3": gw_3_actuals,
    }
    
    # Display Actual information
    
    middle.subheader("Actual VS. Predicted predicted by Game Week")
    
    gw_num_actuals = middle.selectbox(
        "Pick a game week (actual results listed by match id from the table above):",
        ("game_week1",
         "game_week2",
         "game_week3"
         ),  key="gw_num_actuls")
    
    # Get the selected DataFrame and display it
    selected_dataframe = actuals_week_mapping.get(gw_num_actuals)
    if selected_dataframe is not None:
        num_right =len(selected_dataframe.loc[(selected_dataframe["Predicted"] == selected_dataframe["Result"])])
        total = len(selected_dataframe)
        accuracy = num_right/total
        middle.text(f"Predicted correct accuracy: {accuracy:.0%}")        
        middle.dataframe(selected_dataframe, hide_index=True)

    else:
        middle.write(f"DataFrame for {gw_num_actuals} not yet implemented")
        middle.text("Predicted correct accuracy: NONE")

    #############
    # Right column
    #############
    
    # Import table CSV files
    table_1_game_df = pd.read_csv("table_1_game.csv")
    table_2_game_df = pd.read_csv("table_2_game.csv")
    table_3_game_df = pd.read_csv("table_3_game.csv")    
    # Mapping for selecte gameweek to correct table dataframe
    table_mapping = {
        "post game week 1": table_1_game_df,
        "post game week 2": table_2_game_df,
        "post game week 3": table_3_game_df,
        "post game week 4": table_3_game_df,
        "post game week 5": table_3_game_df,
        "post game week 6": table_3_game_df,
        "post game week 7": table_3_game_df,
        "post game week 8": table_3_game_df,
        "post game week 9": table_3_game_df,
        "post game week 10": table_3_game_df,
        "post game week 11": table_3_game_df,
        "post game week 12": table_3_game_df,
        "post game week 13": table_3_game_df,
        "post game week 14": table_3_game_df,
        "post game week 15": table_3_game_df,
        "post game week 16": table_3_game_df,
        "post game week 17": table_3_game_df,
        "post game week 18": table_3_game_df,
        "post game week 19": table_3_game_df,
        "post game week 20": table_3_game_df
    }
    
    # Display pick and dataframe
    right.subheader("Quick Table View ")
    right.text("Note: to see all parts of the Table click on the Tables tab at top of this page")

    gw_num_tables = right.selectbox(
        "Pick a game week:",
        ("post game week 1",
         "post game week 2",
         "post game week 3",
        "post game week 4",
        "post game week 5",
        "post game week 6",
        "post game week 7",
        "post game week 8",
        "post game week 9",
        "post game week 10",
        "post game week 11",
        "post game week 12",
        "post game week 13",
        "post game week 14",
        "post game week 15",
        "post game week 16",
        "post game week 17",
        "post game week 18",
        "post game week 19",
        "post game week 20" 
         ),  key="quick_tables")
    
    # Get the selected DataFrame and display
    selected_dataframe = table_mapping.get(gw_num_tables)
    if selected_dataframe is not None:
        right.dataframe(selected_dataframe, column_order=("Pos","Team"), hide_index=True)

    else:
        right.write(f"DataFrame for {gw_num_tables} not yet implemented")
        

################################################
# Contents of tab2 - Full Table display
################################################

# Convert scrapped table content from Premierleague.com        
with tab2:    
    # Import game week table CSV files
    table_1_game_df = pd.read_csv("table_1_game.csv")
    table_1_game_df.drop(["Unnamed: 0"], axis=1, inplace=True)
    
    table_2_game_df = pd.read_csv("table_2_game.csv")

    table_3_game_df = pd.read_csv("table_3_game.csv")
    
    # Mapping of selected text to proper dataframe
    table_mapping = {
        "post game week 1": table_1_game_df,
        "post game week 2": table_2_game_df,
        "post game week 3": table_3_game_df,
        "post game week 4": table_3_game_df,
        "post game week 5": table_3_game_df,
        "post game week 6": table_3_game_df,
        "post game week 7": table_3_game_df,
        "post game week 8": table_3_game_df,
        "post game week 9": table_3_game_df,
        "post game week 10": table_3_game_df,
        "post game week 11": table_3_game_df,
        "post game week 12": table_3_game_df,
        "post game week 13": table_3_game_df,
        "post game week 14": table_3_game_df,
        "post game week 15": table_3_game_df,
        "post game week 16": table_3_game_df,
        "post game week 17": table_3_game_df,
        "post game week 18": table_3_game_df,
        "post game week 19": table_3_game_df,
        "post game week 20": table_3_game_df
    }

    st.subheader("Complete Table after all matches are played for the selected game week")    

    gw_num_tables = st.selectbox(
        "Pick a game week:",
        ("post game week 1",
         "post game week 2",
         "post game week 3",
        "post game week 4",
        "post game week 5",
        "post game week 6",
        "post game week 7",
        "post game week 8",
        "post game week 9",
        "post game week 10",
        "post game week 11",
        "post game week 12",
        "post game week 13",
        "post game week 14",
        "post game week 15",
        "post game week 16",
        "post game week 17",
        "post game week 18",
        "post game week 19",
        "post game week 20"          
         ),  key="gw_num_tables")
    
    # Get the selected DataFrame and display
    selected_dataframe = table_mapping.get(gw_num_tables)
    if selected_dataframe is not None:
        st.dataframe(selected_dataframe, hide_index=True)

    else:
        middle.write(f"DataFrame for {gw_num_tables} not yet implemented")
    
################################################
# Contents of tab3 - About (contains write up of project)
################################################
        
with tab3:

    # Read and display markdown file
    with open("prediction_web1.md", "r") as file:
        markdown_content = file.read()
    st.markdown(markdown_content)
    
    st.image("confusion_matrix_first_round.png", caption="Random Forest Confusion Matrix - first run")
    
    # Read and display markdown file
    with open("prediction_web2.md", "r") as file:
        markdown_content = file.read()
    st.markdown(markdown_content)    
    
    st.image("confusion_matrix_second_round.png", caption="Random Forest Confusion Matrix - fixed data leakage")
    
    # Read and display markdown file
    with open("prediction_web3.md", "r") as file:
        markdown_content = file.read()
    st.markdown(markdown_content)    
    
    st.image("permutation_summary_chart.png", caption="Feature importance")
   
    # Read and display markdown file
    with open("prediction_web4.md", "r") as file:
        markdown_content = file.read()
    st.markdown(markdown_content)    
    
    
    
    
    
