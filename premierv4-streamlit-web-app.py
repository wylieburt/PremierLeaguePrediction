# import libraries

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime, timedelta
import seaborn as sns
import pickle
import joblib

# Machine learning packages
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Local libraries
import Analytics.Advanced_Analytics as aa
import Plot.timeseries_plots as tp
import Plot.performance_plots as pp
import Plot.table_style as ts
import Data.preprocess as preproc
import Stats.bayesian_analysis as ba

########################
# Import model and data
########################

# 20-26 seaons
clf_reduced = joblib.load('Models/premier_random_forest_20_26_prediction.joblib')
clf_reduced_name = 'premier_random_forest_20_26'
data_for_avg = joblib.load('Data/premier_random_forest_20_26_prediction_data.joblib')

#All matches with results only

win_count_df = data_for_avg.groupby("Team")["Result"].value_counts().reset_index()

# All matches with data, team, opp, and result
all_data_df = pd.read_csv("Data/match_data_20_26.csv")
all_data_df['Result'] = all_data_df['Result'].str.split(' ').str[0]
all_unique = all_data_df[all_data_df['Team'] < all_data_df['Opp']]
analysis_df = all_unique

# Timeseries data for the data info tab
timeseries_df = analysis_df.groupby(["Date", "Result"])["Result"].value_counts().reset_index()
print(timeseries_df.head())
timeseries_df['Date'] = pd.to_datetime(timeseries_df['Date'])

# Extract date components
timeseries_df['Year'] = timeseries_df['Date'].dt.year
timeseries_df['Month'] = timeseries_df['Date'].dt.month
timeseries_df['Day'] = timeseries_df['Date'].dt.day
timeseries_df['DayOfWeek'] = timeseries_df['Date'].dt.dayofweek  # Monday=0, Sunday=6
timeseries_df['DayOfYear'] = timeseries_df['Date'].dt.dayofyear
timeseries_df['Week'] = timeseries_df['Date'].dt.isocalendar().week
timeseries_df['Quarter'] = timeseries_df['Date'].dt.quarter

# Add readable day names
timeseries_df['DayName'] = timeseries_df['Date'].dt.day_name()
timeseries_df['MonthName'] = timeseries_df['Date'].dt.month_name()

columns_to_keep = ['Date','Team', 'Opp', 'Result']
match_result_lookup = all_data_df[columns_to_keep]

enhanced_results_summary_df = preproc.enhanced_results_summary(match_result_lookup)

merged_df = pd.merge(data_for_avg, enhanced_results_summary_df, how = "inner", on = "Team")

accuracy_tracking = pd.DataFrame({"Game Week" : ["GW 1", "GW 2", "GW 3", "GW 4", "GW 5"],
                                  "Accuracy" : [60, 70, 40, 70, 50],
                                  "Running Median" : [60, 65, 60, 65, 60]})

################################################
# Create tabs to display at top of each tab
################################################

tab1, tab2, tab3, tab4 = st.tabs(["Make Prediction", "Data and Model", "About Project", "Analytics"])

################################################
# Contents of tab1 - Make Prediction
################################################
with tab1:

    # Image at top of tab
    st.image("Images/crystal_palace_stadium.jpg")

    # Add title and references
    st.title("Premier League Match Prediction")
    st.text("Data Science and Machine Learning by Wylie")
    st.text("Data provided by FBref.com")

    
    # Add high level instructions
    st.subheader("Select the Home and Away teams, along with their team weights to boost the probability if you want. Click 'Predict'")
    st.text("Note: Team weights are very rarely needed and should range beteen 0 and 2.  The model is predicting a median accuracy of 65% with no weights.")
    
    with st.form("predict_form"):
        col1, col2, col3 = st.columns([1,1,1], vertical_alignment="bottom")
        
        with col1:
            team1_name = st.selectbox(
                "Home Team Name:",
                ("Arsenal", "Aston Villa", 
                 "Bournemouth", "Brentford","Burnley", "Brighton",
                 "Chelsea", "Crystal Palace",
                 "Everton",
                 "Fulham",
                 "Leeds United","Liverpool",
                 "Manchester City", "Manchester Utd",
                 "Newcastle Utd", "Nott'ham Forest", 
                 "Sunderland", "Tottenham", 
                 "West Ham", "Wolves"
                 ), width=150, key="form_team1_name"
            )
            
            team1_weight = st.number_input(
                label = "Home team weight",
                min_value = 0.00,
                max_value = 2.00,
                value = 1.00,
                width=150, key = "form_team1_weight")
            
        with col3:
            team2_name = st.selectbox(
                "Away Team Name:",
                ("Arsenal", "Aston Villa", 
                 "Bournemouth", "Brentford","Burnley", "Brighton",
                 "Chelsea", "Crystal Palace",
                 "Everton",
                 "Fulham",
                 "Leeds United","Liverpool",
                 "Manchester City", "Manchester Utd",
                 "Newcastle Utd", "Nott'ham Forest", 
                 "Sunderland", "Tottenham", 
                 "West Ham", "Wolves"
                 ), width=150, key="form_team2_name"
            )
               
            team2_weight = st.number_input(
                label = "Away team weight",
                min_value = 0.00,
                max_value = 2.00,
                value = 1.00,
                width=150, key = "form_team2_weight")
            
    
        # Every form must have a submit button.
        submitted = st.form_submit_button("Predict")
        if submitted:
            
            # Remove results feature, not needed for creating averages
            data_for_avg.drop(['Result'], axis=1, inplace = True)
            
            # First collect all data for each team in to separate dataframes.
            tmp_team1 = data_for_avg[data_for_avg['Team'] == team1_name].copy()
            tmp_team1.drop(['Team'], axis=1, inplace = True)
        
            tmp_team2 = data_for_avg[data_for_avg['Team'] == team2_name].copy()
            tmp_team2.drop(['Team'], axis=1, inplace = True)
            len(tmp_team1)
            len(tmp_team2)
            

            # if one team has played less matches than the other apply a Baysian Shrinkage
            # to make sure the averages for that team are reliable.  This is more pronounced
            # for the teams that are newer to the premier league.
            
            if (len(tmp_team1) < 50) | (len(tmp_team2) < 50):
                bay_application = 'Applying Bayesien Shrinkage due to the imbalance of the number in matches of each team in historic data'
                
                # Convert DataFrames to list of lists format expected by Bayesian analyzer
                team1_data_list = tmp_team1.values.tolist()  # Convert DataFrame to list of lists
                team2_data_list = tmp_team2.values.tolist()  # Convert DataFrame to list of lists
                
                # Initialize analyzer
                analyzer = ba.BayesianTeamAnalyzer()
                
                # Run Bayesian comparison
                results = analyzer.compare_teams_bayesian(team1_data_list, team2_data_list)
                results_df = pd.DataFrame(results)
                
                summary_df, tmp_team1_mean, tmp_team2_mean = analyzer.summary_table()

            else:
                bay_application = 'Simple mean of historic data for each team'
            
                # Average numerical data in each datafram and put into a new dataframe.    
                averaged_data = tmp_team1.mean().to_frame().T
                tmp_team1_mean = averaged_data.reset_index(drop=True)
            
                averaged_data = tmp_team2.mean().to_frame().T
                tmp_team2_mean = averaged_data.reset_index(drop=True)
        
            # Combine average data dataframes and reset the index
            combined_avg = pd.concat([tmp_team1_mean, tmp_team2_mean], axis=0).reset_index(drop=True)
           
            # Ensure the order of features matches the training data
            #combined_avg =combined_avg[tmp_team1.columns]
            
            # Random Forest prediction
            predict_proba = clf_reduced.predict_proba(combined_avg)
            
            # Extract probabilities
            home_lose, home_tie, home_win = predict_proba[0]
            away_lose, away_tie, away_win = predict_proba[1]
            
            # When Win probabilitie are extremely close, fovor the team with the most wins against the other team historically.
            if (((away_win - home_win) ** 2) * 100) < 0.041:
                favoring = 'Applied do to the closeness of both win probabilites'
                tmp_match_results = match_result_lookup.loc[(match_result_lookup["Team"].isin([team1_name, team2_name])) & (match_result_lookup["Opp"].isin([team1_name, team2_name]))].reset_index()        
                result_summary = tmp_match_results.loc[(tmp_match_results["Result"] == "W")].groupby(["Team"])["Result"].value_counts().reset_index()
                
                if len(result_summary) > 1:
                    if (result_summary.loc[0, "count"] != result_summary.loc[1, "count"]):
                        max_row_index = result_summary['count'].idxmax()
                        team_name_max_wins = result_summary.loc[max_row_index, 'Team']
                        
                        if (team_name_max_wins == team1_name):
                            team1_weight = 2.0
                        elif (team_name_max_wins == team2_name):
                            team2_weight = 2.0
                                   
                    else:
                        max_row_index = result_summary['count'].idxmax()
                        team_name_max_wins = result_summary.loc[max_row_index, 'Team']
                        
                        if (team_name_max_wins == team1_name):
                            team1_weight = 2.0
                        elif (team_name_max_wins == team2_name):
                            team2_weight = 2.0
                        
                else:
                    team_wins = win_count_df.loc[(win_count_df["Team"].isin([team1_name, team2_name])) & (win_count_df["Result"] == "W"),["Team","count"]].reset_index()        
                    max_row_index = team_wins['count'].idxmax()
                    team_name_max_wins = team_wins.loc[max_row_index, 'Team']
                    
                    if (team_name_max_wins == team1_name):
                        team1_weight = 2.0
                    elif (team_name_max_wins == team2_name):
                        team2_weight = 2.0 

                    
                    #issues = 'None'
                    #favoring = 'None applied due to issue'
                    
            else:
                favoring = 'None applied'

            
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
            elif (tie_prob > (home_win_prob * 1.2)) and (tie_prob > (away_win_prob * 1.2)):
                pick = "Tie"
            else:
                pick = max(home_win_prob, away_win_prob)
            
            # Convert to percentage and reduce number of decimals
            home_win_prob = round((home_win_prob * 100),2)
            tie_prob = round((tie_prob * 100),2)
            away_win_prob = round((away_win_prob * 100),2)
        
        
            # Output prediction to tab.  Using markdown to set text color
            
            
            col1.subheader(f"Home Win: {home_win_prob}")

            col3.subheader(f"Away Win: {away_win_prob}")

            col2.subheader(f"Tie Prob: {tie_prob}")

            st.subheader(f"Pick: {pick}")
        
                
            
            st.text("Notes on prediction:")
            st.text(f"Statistical Logic: {bay_application}")
            st.text(f"Actual Probabilities: {predict_proba}")
            st.text(f"Favoring: {favoring} - Home Win: {home_win} Away Win: {away_win}")
            st.text("***Team stats after any statistical logic or favoring in order of importance to prediction***")
            st.dataframe(combined_avg, column_order=("GF", "Long_Cmp", "Poss", "SoT", "Blocks", "Succ"))
            
        bay_application = ''
        favoring = ''
        
    gw_num_pick = st.selectbox(
        "Pick a game week to see matches, results, and predictions:",
        ("game_week1",
         "game_week2",
         "game_week3",
         "game_week4",
         "game_week5"
         ),  key="gw_num_pick")
    
    # Create dataframes for each game week containing Match ID, Actual  Result, Predicted Result
   
    # Game week 1
    gw_1_actuals_list = [["-", 1,"Liverpool", "Bournemouth", "4-2", "Home Win", "Home Win"],
                      ["-", 2,"Aston Villa", "Newcastle", "0-0", "Tie", "Home Win"],
                      ["-", 3,"Brighton", "Fulham", "1-1", "Tie", "Tie"],
                      ["-", 4,"Sunderland", "West Ham", "3-0", "Home Win", "Home Win"],
                      ["-", 5,"Spurs", "Burnley", "3-0", "Home Win", "Home Win"],
                      ["-", 6,"Wolves", "Man City", "0-4", "Away Win", "Away Win"],
                      ["-", 7,"Nott'm Forest", "Brentford", "3-1", "Home Win", "Tie"], 
                      ["-", 8,"Chelsea", "Crystal Palace", "0-0", "Tie", "Home Win"],
                      ["-", 9,"Man Utd", "Arsenal", "0-1", "Away Win", "Away Win"],
                      ["-", 10,"Leeds United", "Everton", "1-0", "Home Win", "Tie"]]
    gw_1_actuals = pd.DataFrame(gw_1_actuals_list, columns=["Date", "match ID","Home","Away", "Score", "Result", "Predicted"])

    # Game week 2    
    gw_2_actuals_list = [["-", 1,"West Ham", "Chelsea", "1-5", "Away Win", "Away Win"],
                      ["-", 2,"Man City", "Spurs", "0-2", "Away Win", "Home Win"],
                      ["-", 3,"Bournemouth", "Wolves", "1-0", "Home Win", "Home Win"],
                      ["-", 4,"Brentford", "Aston Villa", "1-0", "Home Win", "Away Win"],
                      ["-", 5,"Burnley", "Sunderland", "2-0", "Home Win", "Home Win"],
                      ["-", 6,"Arsenal", "Leeds United", "5-0", "Home Win", "Home Win"],
                      ["-", 7,"Crystal Palace", "Nott'm Forest", "1-1", "Tie", "Tie"], 
                      ["-", 8,"Everton", "Brighton", "2-0", "Home Win", "Home Win"],
                      ["-", 9,"Fulham", "Man Utd", "1-1", "Tie", "Away Win"],
                      ["-", 10,"Newcastle", "Liverpool", "2-3", "Away Win", "Away Win"]]
    gw_2_actuals = pd.DataFrame(gw_2_actuals_list, columns=["Date", "match ID","Home","Away", "Score", "Result", "Predicted"])

    # game week 3
    gw_3_actuals_list = [["-", 1,"Chelsea", "Fulham", "2-0", "Home Win", "Home Win"],
                      ["-", 2, "Man Utd", "Burnley", "3-2", "Home Win", "Home Win"],
                      ["-", 3, "Sunderland", "Brentford", "2-1", "Home Win", "Home Win"],
                      ["-", 4, "Spurs", "Bournemouth", "0-1", "Away Win", "Home Win"],
                      ["-", 5, "Wolves", "Everton", "2-3", "Away Win", "Home Win"],
                      ["-", 6, "Leeds United", "Newcastle", "0-0", "Tie", "Away Win"],
                      ["-", 7, "Brighton", "Man City", "2-1", "Home Win", "Away Win"], 
                      ["-", 8, "Nott'm Forest", "West Ham", "0-3", "Away Win", "Tie"],
                      ["-", 9, "Liverpool", "Arsenal", "1-0", "Home Win", "Home Win"],
                      ["-", 10,"Aston Villa", "Crystal Palace", "0-3", "Away Win", "Home Win"]]
    gw_3_actuals = pd.DataFrame(gw_3_actuals_list, columns=["Date", "match ID","Home","Away", "Score", "Result", "Predicted"])

    # game week 4
    gw_4_actuals_list = [["-", 1,"Arsenal", "Nott'm Forest", "3-0", "Home Win", "Home Win"],
                    ["-", 2,"Bournemouth", "Brighton", "2-1", "Home Win", "Away Win"],
                    ["-", 3,"Crystal Palace", "Sunderland", "0-0", "Tie", "Tie"],
                    ["-", 4,"Everton", "Aston Villa", "0-0", "Tie", "Away Win"],
                    ["-", 5,"Fulham", "Leeds United", "1-0", "Home Win", "Home Win"],
                    ["-", 6,"Newcastle", "Wolves", "1-0", "Home Win", "Home Win"],
                    ["-", 7,"West Ham", "Spurs", "0-3", "Away Win", "Away Win"],
                    ["-", 8,"Brentford", "Chelsea", "2-2", "Tie", "Away Win"],
                    ["-", 9,"Burnley", "Liverpool", "0-1", "Away Win", "Away Win"],
                    ["-", 10,"Man City", "Man Utd", "3-0", "Home Win", "Home Win"]]
    gw_4_actuals = pd.DataFrame(gw_4_actuals_list, columns=["Date", "match ID","Home","Away", "Score", "Result", "Predicted"])
    
    # game week 5
    gw_5_actuals_list = [["Sat 20 Sep 04:30","Liverpool", "Everton", "2-1", "Home Win", "Home Win"],
                      ["Sat 20 Sep 07:00", "Brighton", "Spurs", "2-2", "Tie", "Away Win"],
                      ["Sat 20 Sep 07:00", "Burnley", "Nott'm Forest", "1-1", "Tie", "Tie"],
                      ["Sat 20 Sep 07:00", "West Ham", "Crystal Palace", "1-2",  "Away Win", "Away Win"],
                      ["Sat 20 Sep 07:00", "Wolves", "Leeds Utd", "1-3",  "Away Win", "Away Win"],
                      ["Sat 20 Sep 09:30", "Man Utd", "Chelsea", "2-1",  "Home Win", "Away Win"],
                      ["Sat 20 Sep 12:00", "Fulham", "Brentford", "3-1",  "Home Win", "Home Win"], 
                      ["Sun 21 Sep 06:00", "Bournemouth", "Newcastle", "0-0",  "Tie", "Away Win"],
                      ["Sun 21 Sep 06:00", "Sunderland", "Aston Villa", "1-1",  "Tie", "Away Win"],
                      ["Sun 21 Sep 08:30", "Arsenal", "Man City", "1-1",  "Tie", "Away Win"]]
    gw_5_actuals = pd.DataFrame(gw_5_actuals_list, columns=["Date", "Home","Away", "Score", "Result", "Predicted"])
  
    # mapping of game selection text to the correct dataframe
    actuals_week_mapping = {
        "game_week1": gw_1_actuals,
        "game_week2": gw_2_actuals,
        "game_week3": gw_3_actuals,
        "game_week4": gw_4_actuals, 
        "game_week5": gw_5_actuals
    }
    
    # Display Actual information
    
    st.subheader("Schedule with Actual VS. Predicted")
    
    # Get the selected DataFrame and display it
    selected_dataframe = actuals_week_mapping.get(gw_num_pick)
    if selected_dataframe is not None:
        if  selected_dataframe.isna().sum().sum() > 0:
            predict_calc_df = selected_dataframe.dropna(how = "any" )
        else:
            predict_calc_df = selected_dataframe
        num_right =len(predict_calc_df.loc[(predict_calc_df["Predicted"] == predict_calc_df["Result"])])
        total = len(predict_calc_df)
        if total > 0:
            accuracy = num_right/total
            styled_df = selected_dataframe.style.apply(ts.highlight_multiple_conditions, axis=1)        
            st.dataframe(styled_df, column_order= ("Date", "Home","Away", "Score", "Predicted"), hide_index=True)
            st.text(f"Predicted correct accuracy: {num_right} of {total} played - {accuracy:.0%}")
        else:
            styled_df = selected_dataframe.style.apply(ts.highlight_multiple_conditions, axis=1)        
            st.dataframe(styled_df, column_order= ("Date","Home","Away", "Score", "Predicted"), hide_index=True)
        
    else:
        st.write(f"DataFrame for {gw_num_pick} not yet implemented")
        st.text("Predicted correct accuracy: NONE")
           
    # Import table CSV file with all tables in it
    table_all_df = pd.read_csv("tables_all.csv")
    
    # create a dataframe for each game week from table_all_df and selecting  on gw_num 
    table_1_game_df = table_all_df[table_all_df["gw_num"] == 1]
    table_2_game_df = table_all_df[table_all_df["gw_num"] == 2]
    table_3_game_df = table_all_df[table_all_df["gw_num"] == 3] 
    table_4_game_df = table_all_df[table_all_df["gw_num"] == 4] 
    table_5_game_df = table_all_df[table_all_df["Pl"] == 5] 
    
    # Mapping for selecte gameweek to correct table dataframe
    table_mapping = {
        "post game week 1": table_1_game_df,
        "post game week 2": table_2_game_df,
        "post game week 3": table_3_game_df,
        "post game week 4": table_4_game_df,
        "post game week 5": table_5_game_df,
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
    st.subheader("Table View ")
    st.text("Note: Table is updated on Sunday evening of each game week.")

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
         "post game week 20"),  key="full_tables")
    
    # Get the selected DataFrame and display
    selected_dataframe = table_mapping.get(gw_num_tables)
    if selected_dataframe is not None:
        st.dataframe(selected_dataframe, column_order=("Pos","Team","Pl","W","D","L","GF","GA","GD","Pts"), hide_index=True)

    else:
        st.write(f"DataFrame for {gw_num_tables} not yet implemented")
        

################################################
# Contents of tab2 - Full Table display
################################################

      
with tab2:  
    
  
    
    st.header("Interesting views of the historic data used in this model")
    
    st.subheader("Basic information on dataset")
    st.text("Date Range--> 9/12/2020 and 8/31/2025")
    st.text(f"Total number of matches used for training and testing--> {data_for_avg.shape[0] / 2}")
    st.text(f"Each team in a match is represented with a row in the dataset for total samples--> {data_for_avg.shape[0]}")
    st.text(f"Total teams in historic dataset--> {len(enhanced_results_summary_df)}")
    
    st.text("Sample of data used for training and testing")
    st.dataframe(data_for_avg.sample(frac=0.003), hide_index=True)
    
    st.subheader("Matches played by each team in the historic dataset")    

    games_played_df = data_for_avg['Team'].value_counts().reset_index()
    st.dataframe(games_played_df, width= 175, hide_index=True)
    
    st.subheader("Most Points - Ordered by Points")    
    st.dataframe(enhanced_results_summary_df, hide_index=True)
    
    #######################
    # Performance
    #######################
    # Performance dashboard
    st.subheader("Performance Dashboard")
    dashboard_fig = pp.create_football_performance_dashboard(timeseries_df)
    st.pyplot(dashboard_fig)
    
    # Team performance
    st.subheader("Team Performance Chart")
    team_performance_fig =  pp.team_performance(merged_df)
    st.pyplot(team_performance_fig)

    ######################
    # TIMESERIES PLOTS
    ######################
    with st.form("timeseries_plot_form"):
        st.write("Generate plots to explore win rates over time in heatmaps")
        plot_name = st.selectbox(
            "Timeseies plot type:",
            ("Heatmap Analysis",
             "Simple",
             "Monthly Summary"
             ), width=150, key="form_timeseries_plot_name")
    
        # Every form must have a submit button.
        submitted = st.form_submit_button("Generate Plot")
        if submitted:
            if plot_name == "Heatmap Analysis":
                heatmaps_fig = tp.create_heatmaps(timeseries_df)
                st.pyplot(heatmaps_fig)
            elif plot_name == "Simple":    
                simple_plt = tp.simple_win_rate_heatmap(timeseries_df)
                st.pyplot(plt)
            elif plot_name == "Monthly Summary":
                monthly_plt = tp.monthly_summary_heatmap(timeseries_df)
                st.pyplot(monthly_plt)
    
    
    st.subheader("Model Description")
    st.text("Model: Random Forest Classifer")
    
    
    st.subheader("Model Performance")
    
    st.line_chart(accuracy_tracking, x="Game Week", color=["#0000ff", "#66ff33"])
    
    st.subheader("Model Description")
    

    # Create DataFrame using dictionary
    tmp_df = pd.DataFrame([clf_reduced.feature_importances_], columns=clf_reduced.feature_names_in_)

    
    st.text("Model type: Random Forest Classifier")
    st.text(f"Number of samples used for training: {clf_reduced._n_samples}")
    st.text(f"Number of Decision Trees in the forest: {clf_reduced.n_estimators}")
    st.text(f"Maxium depth or levels of splits in decision trees: {clf_reduced.max_depth}")
    st.text(f"Maxium features or number of features to consider when making each split in each decision tree: {clf_reduced.max_features}")
    st.text(f"Number of pridiction possibility: {clf_reduced.n_classes_}")
    st.text("Original number of features considered before permutation importance: 70")
    st.text("Feature Importances from Model Assessment:")
    st.dataframe(tmp_df)
    
    
    
################################################
# Contents of tab3 - About (contains write up of project)
################################################
        
with tab3:

    # Read and display markdown file
    with open("Markdown/prediction_web1.md", "r") as file:
        markdown_content = file.read()
    st.markdown(markdown_content)
    
    st.image("Images/confusion_matrix_first_round.png", caption="Random Forest Confusion Matrix - first run")
    
    # Read and display markdown file
    with open("Markdown/prediction_web2.md", "r") as file:
        markdown_content = file.read()
    st.markdown(markdown_content)    
    
    st.image("Images/confusion_matrix_second_round.png", caption="Random Forest Confusion Matrix - fixed data leakage")
    
    # Read and display markdown file
    with open("Markdown/prediction_web3.md", "r") as file:
        markdown_content = file.read()
    st.markdown(markdown_content)    
    
    st.image("Images/permutation_summary_chart.png", caption="Feature importance")
   
    # Read and display markdown file
    with open("Markdown/prediction_web4.md", "r") as file:
        markdown_content = file.read()
    st.markdown(markdown_content)    
    
################################################
# Contents of tab3 - Analytics
################################################
        
with tab4:
    
    football_df = timeseries_df.reset_index()
    football_df['Date'] = pd.to_datetime(football_df['Date'])
    streak_fig, momentum_fig, predictive_fig, stats_fig, cyclical_fig, performance_fig, volatility_fig, scenario_fig = aa.advanced_football_analytics_suite(football_df)
       
    st.pyplot(streak_fig)
