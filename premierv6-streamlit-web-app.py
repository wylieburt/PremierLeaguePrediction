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
import pydeck as pdk
import plotly.express as px

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
import Map_code.hub_spoke_map as hub
import Data.stadiums_merge as sm

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

accuracy_tracking = pd.DataFrame({"Game Week" : ["GW 1", "GW 2", "GW 3", "GW 4", "GW 5", "GW 6"],
                                  "Accuracy" : [60, 70, 40, 70, 50, 40],
                                  "Running Median" : [60, 65, 60, 65, 60, 55]})

#Stadium data

stadium_data = pd.read_csv("Data/stadiums.csv")

all_df = pd.DataFrame(all_data_df["Team"].unique(), columns = ["Team"])
idx = all_df[all_df['Team'] == "Nott'ham Forest"].index
all_df.loc[idx, 'Team'] = "Nottingham Forest"

stadiums_pl = pd.merge(all_df, stadium_data, how="left", on="Team")
stadiums_pl.info()

# Player data used in tab6 - Players
all_players_df = pd.read_csv("Data/players_25_26.csv")
all_players_df.drop(["Season", "Comp", "-9999"], axis=1, inplace=True)

country_codes_df = pd.read_csv("Data/world.csv")

all_players_df['Nation'] = all_players_df['Nation'].str.replace(r'[A-Z]{3}', '', regex=True).str.strip()

all_players_df['Nation'] = all_players_df['Nation'].replace({'eng': 'gb',
                                                             'wls': 'gb',
                                                             'nir': 'gb',
                                                             'sct': 'gb'})

# Add flag_path URLs
all_players_df['flag_path'] = all_players_df['Nation'].apply(
    lambda x: f"https://flagcdn.com/64x48/{x}.png"
)

all_players_df['Nation'] = all_players_df['Nation'].str.upper()

# Add extra stadium data
stadiums_pl = sm.stadium_merge(stadiums_pl)

# Grab badge from stadiums-pl and add to players
all_players_df['Team'] = all_players_df['Team'].replace({"Nott'ham Forest": 'Nottingham Forest'})
all_players_df = all_players_df.merge(stadiums_pl[["Team", "Badge"]], how="left", on="Team")

# Add in a % of total minutes for the season 28*90
all_players_df["Perc_Min_Season"] = all_players_df["Min"] / (28 * 90)
all_players_df.fillna(value = 0, inplace=True)    

# Import country location data
countries_df = pd.read_csv("Data/countries.csv")

all_players_df = all_players_df.merge(countries_df[['country', 'latitude', 'longitude', 'name']], how="left", left_on='Nation', right_on='country')


tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Make Prediction", "Data and Model", "About Project", "Analytics", "Maps", "Players", "Leader Boards"])

# Track which tab is active (this is a workaround since Streamlit doesn't directly expose active tab)
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Tab 1"

################################################
# Contents of tab1 - Make Prediction
################################################
with tab1:
    st.session_state.active_tab = "Tab 1"
    st.sidebar.empty()
    # Image at top of tab
    st.image("Images/crystal_palace_stadium.jpg")

    # Add title and references
    st.title("Premier League Match Prediction")
    st.text("Data Science and Machine Learning by Wylie")
    st.text("Inspired by David Burt, Camille Burt, and Maddie Burt")
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
            st.write("Expand the sidebar to the left to see prediction.")
            # Remove results feature, not needed for creating averages
            data_for_avg.drop(['Result'], axis=1, inplace = True)
            
            # First collect all data for each team in to separate dataframes.
            tmp_team1 = data_for_avg[data_for_avg['Team'] == team1_name].copy()
            tmp_team1.drop(['Team'], axis=1, inplace = True)
            tmp_team1_mean_disp = tmp_team1.mean().to_frame().T
            
            tmp_team2 = data_for_avg[data_for_avg['Team'] == team2_name].copy()
            tmp_team2.drop(['Team'], axis=1, inplace = True)
            tmp_team2_mean_disp = tmp_team2.mean().to_frame().T
            
            original_combined_avg_disp = pd.concat([tmp_team1_mean_disp, tmp_team2_mean_disp], axis=0).reset_index(drop=True)
            original_combined_avg_disp["Team"] = [team1_name, team2_name]


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
            predict_proba_disp = pd.DataFrame(predict_proba, columns=["Loss", "Tie", "Win"])
            predict_proba_disp["Team"] = [team1_name, team2_name]
            
            combined_avg_disp = combined_avg
            combined_avg_disp["Team"] = [team1_name, team2_name]
            
            with st.sidebar:
                st.sidebar.empty()
                st.header("Model predictions of the selected match")
                st.subheader(f"Home Win: {home_win_prob}")
                st.subheader(f"Tie Prob: {tie_prob}")    
                st.subheader(f"Away Win: {away_win_prob}")
    
                st.subheader(f"Pick: {pick}")
            
                st.write("__Notes on prediction:__")
                st.write(f"__Statistical Logic:__ {bay_application}")
                st.write(f"__Favoring:__ {favoring} \n {team1_name}: {home_win} \n {team2_name}: {away_win}")             
                
                st.write(f"{team1_name} was given a weight of {team1_weight}")
                st.write(f"{team2_name} was given a weight of {team2_weight}")

                st.write("Mean average of each team __before__ any processing:")
                
                df_transposed = original_combined_avg_disp.transpose()


                # Set the first row (index 0, which is 'Team') as column names
                df_transposed.columns = df_transposed.iloc[6]
               
                # Drop the 'Team' row since it's now the column names
                df_transposed = df_transposed.drop(df_transposed.index[6])


                df_transposed["Diff"] = df_transposed[team1_name] - df_transposed[team2_name]
                
                styled_df = df_transposed.style.apply(ts.highlight_max, axis=1)
                st.dataframe(styled_df)
                #st.dataframe(df_transposed)
                
                st.text("Team stats __after__ any statistical logic or favoring in order of importance to prediction")
                
                df_transposed = combined_avg.transpose()

                # Set the first row (index 0, which is 'Team') as column names
                df_transposed.columns = df_transposed.iloc[6]
               
                # Drop the 'Team' row since it's now the column names
                df_transposed = df_transposed.drop(df_transposed.index[6])

                df_transposed["Diff"] = df_transposed[team1_name] - df_transposed[team2_name]
                styled_df = df_transposed.style.apply(ts.highlight_max, axis=1)
                st.dataframe(styled_df)
                
               # Get the statistics (row index)
                stats = df_transposed.index.tolist()  # ['SoT', 'GF', 'Poss', 'Long_Cmp', 'Succ', 'Blocks']
                
                # Get team names (excluding 'Diff' column)
                teams = [col for col in df_transposed.columns if col != 'Diff']  # ['Team 1', 'Team 2']
                
                x = np.arange(len(stats))  # the label locations (one for each stat)
                width = 0.35  # the width of the bars
                
                fig, ax = plt.subplots(layout='constrained', figsize=(10, 6))
                
                # Create bars for each team
                for i, team in enumerate(teams):
                    offset = width * i
                    rects = ax.bar(x + offset, df_transposed[team], width, label=team)
                
                # Add some text for labels, title and custom x-axis tick labels, etc.
                ax.set_ylabel('Value of Stat')
                ax.set_title('Stat Comparison by Team')
                ax.set_xticks(x + width/2)  # Center the labels between the two bars
                ax.set_xticklabels(stats)  # Use the stat names as labels
                ax.legend(loc='upper right')
                ax.set_ylim(0, max(df_transposed[teams].max()) * 1.1)  # Dynamic y-limit
                
                st.pyplot(fig)
                
                
                stats_sum_diffs = df_transposed["Diff"].sum()
                
                df_transposed["team1_perc"] = (df_transposed[team1_name] / (df_transposed[team1_name] + df_transposed[team2_name])) * 100
                df_transposed["team2_perc"] = (df_transposed[team2_name] / (df_transposed[team1_name] + df_transposed[team2_name])) * 100
                
               # Get the statistics (row index)
                stats = df_transposed.index.tolist()  # ['SoT', 'GF', 'Poss', 'Long_Cmp', 'Succ', 'Blocks']
                
                # Get the percentage data
                team1_pct = df_transposed['team1_perc']
                team2_pct = df_transposed['team2_perc']
                
                y = np.arange(len(stats))  # the label locations (one for each stat)
                height = 0.5  # the height of the bars
                
                fig, ax = plt.subplots(layout='constrained', figsize=(10, 6))
                
                # Create horizontal stacked bars
                ax.barh(y, team1_pct, height, label=team1_name, color='royalblue')
                ax.barh(y, team2_pct, height, left=team1_pct, label=team2_name, color='mediumseagreen')
                
                # Add some text for labels, title and custom y-axis tick labels, etc.
                ax.set_xlabel('Percentage (%)')
                ax.set_title('Stat Distribution by Team')
                ax.set_yticks(y)
                ax.set_yticklabels(stats)
                ax.legend(loc='upper right', ncol = 2)
                ax.set_xlim(0, 100)  # Percentages go from 0 to 100
                
                # Optional: Add a vertical line at 50% for reference
                ax.axvline(x=50, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
                
                st.pyplot(fig)
                
                st.write(f"The sum of all the differences: {stats_sum_diffs}")
                #st.write(f"A positive sum of differences is a leaning towards {team1_name}")
                
                st.text("Probabilities before statistical logic or favoring:")
                #st.dataframe(predict_proba_disp, column_order=("Team", "Win", "Tie", "Loss"), hide_index=True)
            

        bay_application = ''
        favoring = ''
        
    gw_num_pick = st.selectbox(
        "Pick a game week to see matches, results, and predictions:",
        ("game_week1",
         "game_week2",
         "game_week3",
         "game_week4",
         "game_week5",
         "game_week6",
         "game_week7"
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

    # game week 6
    gw_6_actuals_list = [["Sat 27 Sep 04:30","Brentford", "Man Utd", "3-1", "Home Win", "Away Win"],
                      ["Sat 27 Sep 07:00", "Chelsea", "Brighton",  "1-3", "Away Win", "Home Win"],
                      ["Sat 27 Sep 07:00", "Crystal Palace", "Liverpool", "2-1", "Home Win", "Away Win"],
                      ["Sat 27 Sep 07:00", "Leeds Utd", "Bournemouth",  "2-2", "Tie", "Tie"],
                      ["Sat 27 Sep 07:00", "Man City", "Burnley",  "5-1", "Home Win", "Home Win"],
                      ["Sat 27 Sep 09:30", "Nott'm Forest", "Sunderland",  "0-1", "Away Win", "Tie"],
                      ["Sat 27 Sep 12:00", "Spurs", "Wolves",  "1-1", "Tie", "Home Win"], 
                      ["Sun 28 Sep 06:00", "Aston Villa", "Fulham",  "3-1", "Home Win", "Home Win"],
                      ["Sun 28 Sep 08:30", "Newcastle", "Arsenal",  "1-2", "Away Win", "Away Win"],
                      ["Mon 29 Sep 12:00", "Everton", "West Ham",  "1-1", "Tie", "Away Win"]]
    gw_6_actuals = pd.DataFrame(gw_6_actuals_list, columns=["Date", "Home","Away", "Score", "Result", "Predicted"])
    
    # game week 7
    gw_7_actuals_list = [["Sat 27 Sep 04:30","Bournwmouth", "Fulham", np.nan, np.nan, "Tie"],
                      ["Fri 03 Oct 12:00", "Leeds", "Spurs",  np.nan, np.nan, "Away Win"],
                      ["Sat 04 Oct 07:00", "Arsenal", "West Ham", np.nan, np.nan, "Home Win"],
                      ["Sat 04 Oct 07:00", "Man Utd", "Sunderland",  np.nan, np.nan, "Away Win"],
                      ["Sat 04 Oct 09:30", "Chelsea", "Liverpool",  np.nan, np.nan, "Away Win"],
                      ["Sun 05 Oct 06:00", "Aston Villa", "Burnley",  np.nan, np.nan, "Home Win"],
                      ["Sun 05 Oct 06:00", "Everton", "Crystal Palace",  np.nan, np.nan, "Tie"], 
                      ["Sun 05 Oct 06:00", "Newcastle", "Nott'm Forest",  np.nan, np.nan, "Home Win"],
                      ["Sun 05 Oct 06:00", "Wolves", "Brighton",  np.nan, np.nan, "Away Win"],
                      ["Sun 05 Oct 08:30", "Brentford", "Man City",  np.nan, np.nan, "Away Win"]]
    gw_7_actuals = pd.DataFrame(gw_7_actuals_list, columns=["Date", "Home","Away", "Score", "Result", "Predicted"])
    
    # Only use to post special notes about matches.  Otherwise keep False.
    include_note = True
    gw_note = "game_week7"
    note = "The Manchester United V Sunderland match will be close with Sunderland possibly pulling out a Win.  \n Prediction is a Home Win, but I am over rulling and predicting a Sunderland Win."

    # mapping of game selection text to the correct dataframe
    actuals_week_mapping = {
        "game_week1": gw_1_actuals,
        "game_week2": gw_2_actuals,
        "game_week3": gw_3_actuals,
        "game_week4": gw_4_actuals, 
        "game_week5": gw_5_actuals,
        "game_week6": gw_6_actuals,
        "game_week7": gw_7_actuals
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
            if (include_note and gw_note == gw_num_pick):
                st.text(f"***Special Note on Predictions:*** \n For {gw_num_pick}: {note}")
            st.text(f"Predicted correct accuracy: {num_right} of {total} played - {accuracy:.0%}")
            st.write("**Accuracy Tracking**")
            accuracy_tracking_chart = accuracy_tracking.set_index("Game Week")
            st.line_chart(accuracy_tracking_chart)
        else:
            styled_df = selected_dataframe.style.apply(ts.highlight_multiple_conditions, axis=1)        
            st.dataframe(styled_df, column_order= ("Date","Home","Away", "Score", "Predicted"), hide_index=True)
            st.text("Predicted correct accuracy: NONE")
            st.write(f"Scoreing for {gw_num_pick} matches has not yet been recorded")            
            if (include_note and gw_note == gw_num_pick):
                st.text(f"***Special Note on Predictions:*** \n For {gw_num_pick}: {note}")
    else:
        st.write(f"DataFrame for {gw_num_pick} not yet implemented")
        st.text("Predicted correct accuracy: NONE")
           
    # Import table CSV file with all tables in it
    table_all_df = pd.read_csv("Data/tables_all.csv")
    
    # create a dataframe for each game week from table_all_df and selecting  on gw_num 
    table_1_game_df = table_all_df[table_all_df["gw_num"] == 1]
    table_2_game_df = table_all_df[table_all_df["gw_num"] == 2]
    table_3_game_df = table_all_df[table_all_df["gw_num"] == 3] 
    table_4_game_df = table_all_df[table_all_df["gw_num"] == 4] 
    table_5_game_df = table_all_df[table_all_df["Pl"] == 5]
    table_6_game_df = table_all_df[table_all_df["Pl"] == 6]    
    
    # Mapping for selecte gameweek to correct table dataframe
    table_mapping = {
        "post game week 1": table_1_game_df,
        "post game week 2": table_2_game_df,
        "post game week 3": table_3_game_df,
        "post game week 4": table_4_game_df,
        "post game week 5": table_5_game_df,
        "post game week 6": table_6_game_df
        # "post game week 7": table_6_game_df,
        # "post game week 8": table_6_game_df,
        # "post game week 9": table_6_game_df,
        # "post game week 10": table_6_game_df,
        # "post game week 11": table_3_game_df,
        # "post game week 12": table_3_game_df,
        # "post game week 13": table_3_game_df,
        # "post game week 14": table_3_game_df,
        # "post game week 15": table_3_game_df,
        # "post game week 16": table_3_game_df,
        # "post game week 17": table_3_game_df,
        # "post game week 18": table_3_game_df,
        # "post game week 19": table_3_game_df,
        # "post game week 20": table_3_game_df
    }
    
    # Display pick and dataframe
    st.subheader("Table View ")
    st.text("Note: Table is updated on Sunday evening of each game week.")

    gw_num_tables = st.selectbox(
        "Pick a game week for the Table:",
        ("post game week 1",
         "post game week 2",
         "post game week 3",
         "post game week 4",
         "post game week 5",
         "post game week 6",
         # "post game week 7",
         # "post game week 8",
         # "post game week 9",
         # "post game week 10",
         # "post game week 11",
         # "post game week 12",
         # "post game week 13",
         # "post game week 14",
         # "post game week 15",
         # "post game week 16",
         # "post game week 17",
         # "post game week 18",
         # "post game week 19",
         # "post game week 20"
         ),  key="full_tables")
    
    #Compare teams on the table    
    options = st.multiselect(
    "Optionally, after selecting the game week for the table, you may select teams to compare from that table:",
    ["Arsenal", "Aston Villa", 
     "AFC Bournemouth", "Brentford","Burnley", "Brighton & Hove Albion",
     "Chelsea", "Crystal Palace",
     "Everton",
     "Fulham",
     "Leeds United","Liverpool",
     "Manchester City", "Manchester United",
     "Newcastle United", "Nottingham Forest", 
     "Sunderland", "Tottenham Hotspur", 
     "West Ham United", "Wolverhampton Wonderers"],
    )
    
    # Get the selected DataFrame and display
    selected_dataframe = table_mapping.get(gw_num_tables)
    if selected_dataframe is not None:
        st.dataframe(selected_dataframe, column_order=("Pos","Team","Pl","W","D","L","GF","GA","GD","Pts"), hide_index=True)
        table_comp_df = selected_dataframe.loc[selected_dataframe["Team"].isin(options)]

    else:
        st.write(f"DataFrame for {gw_num_tables} not yet implemented")
    
    st.write("__Team Table Comparison__")
    st.dataframe(table_comp_df, column_order=("Pos","Team","Pl","W","D","L","GF","GA","GD","Pts"), hide_index=True)
                        
################################################
# Contents of tab2 - Data and Model
################################################

      
with tab2:  
    
  
    st.session_state.active_tab = "Tab 2"
    st.sidebar.empty()
    
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
    tab1, tab2 = st.tabs(["Chart", "Dataframe"])
    tab1.pyplot(team_performance_fig)
    tab2.dataframe(merged_df, height=250, use_container_width=True)

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
    st.session_state.active_tab = "Tab 3"
    st.sidebar.empty()
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
# Contents of tab4 - Analytics
################################################
        
with tab4:
    st.session_state.active_tab = "Tab 4"
    st.sidebar.empty()
    football_df = timeseries_df.reset_index()
    football_df['Date'] = pd.to_datetime(football_df['Date'])
    streak_fig, momentum_fig, predictive_fig, stats_fig, cyclical_fig, performance_fig, volatility_fig, scenario_fig = aa.advanced_football_analytics_suite(football_df)
       
    st.pyplot(streak_fig)

################################################
# Contents of tab5 - Maps
################################################

with tab5:
    
    stadium_map, hub_map = st.tabs(["Stadium Map", "Hub and Spoke"])
    
    with stadium_map:
        st.write("A tribute to my Cartographic friends 🌍 ")
        #col1, col2 = st.columns([2,2], vertical_alignment="bottom")
        
        #with col1:
        team_options = ['All Teams'] + sorted(stadiums_pl['Team'].tolist())
        selected_team_name = st.selectbox('Select a team to zoom to:', team_options)
        
        # Determine view state and whether to show points based on selection
        if selected_team_name == 'All Teams':
            # Show all stadiums
            view_state = pdk.ViewState(
                latitude=stadiums_pl['Latitude'].mean(),
                longitude=stadiums_pl['Longitude'].mean(),
                zoom=5,
                pitch=0,
            )
            show_points = True  # Show points when viewing all teams
        else:
            # Zoom to selected team's stadium
            stadium_row = stadiums_pl[stadiums_pl['Team'] == selected_team_name].iloc[0]
            view_state = pdk.ViewState(
                latitude=stadium_row['Latitude'],
                longitude=stadium_row['Longitude'],
                zoom=15,  # High zoom for individual stadium
                pitch=0,
            )
            show_points = False  # Hide points when viewing individual stadium
            #selected_stadium = stadiums_pl[stadiums_pl['Team'] == selected_team].iloc[0]                   
            image_url = stadiums_pl.loc[stadiums_pl["Team"] == selected_team_name]['Stadium Image'].iloc[0]
            badge_url = stadiums_pl.loc[stadiums_pl["Team"] == selected_team_name]['Badge']#.iloc[0]
            team_url = stadiums_pl.loc[stadiums_pl["Team"] == selected_team_name]['URL'].values[0]
            stadium_url = stadiums_pl.loc[stadiums_pl["Team"] == selected_team_name]['Stadium URL'].values[0]
            
            st.header(f"{selected_team_name} plays here at {stadium_row['Name']}")
            
            html_content = f'''
            <a href="{image_url}" target="_blank" rel="noopener noreferrer">
                <img src="{image_url}" 
                     style="width: 200%; max-width: 680px; border-radius: 10px; margin-bottom: 15px; cursor: pointer; transition: transform 0.2s ease;" 
                     onmouseover="this.style.transform='scale(1.05)'" 
                     onmouseout="this.style.transform='scale(1.0)'"
                     title="Click to view full size image">
            </a>
            '''
            st.html(html_content)
        
            st.header(f"Stadium capacity: \n {stadium_row['Capacity']}")
            st.header(f"Site for the team that playes here: \n {team_url}")
            st.header(f"Site for the stadium: \n {stadium_url}")
         
            st.header(f"local map of {stadium_row['Name']} and surrounding area")
        
        #Create layers conditionally
        layers = []
        if show_points:
            layers.append(
                pdk.Layer(
                    'ScatterplotLayer',
                    data=stadiums_pl,
                    get_position='[Longitude, Latitude]',
                    get_color='[0, 255, 0]',
                    get_radius=800,
                    radius_scale=1,
                    radius_min_pixels=5,
                    radius_max_pixels=50,
                    pickable=True
                )
            )
            
        st.pydeck_chart(pdk.Deck(
            map_style='road',
            initial_view_state=view_state,
            layers=layers,
            tooltip={
                'html': '''
                <div style="background-color: rgba(0,0,0,0.9); padding: 12px; border-radius: 6px;">
                    <h3 style="color: #ff8c00; margin: 0 0 5px 0;">{Team}</h3>
                    <p style="color: white; margin: 2px 0; font-size: 14px;">{Name}</p>
                    <p style="color: white; margin: 2px 0;">Capacity: {Capacity}</p>
                </div>
                '''
            }
        ))
                          
    with hub_map:
        with st.form("hub_form"):
            # Describe a hub and spoke map
            
            with open("Markdown/hub_spoke_map_desc.md", "r") as file:
                markdown_content = file.read()
            st.markdown(markdown_content) 
            
            # Create selectbox for team selection
            team_options = ['All Teams'] + sorted(stadiums_pl['Team'].tolist())
            selected_team = st.selectbox('Select a team to show the hub and spoke map for that team:', team_options)
            submitted = st.form_submit_button("Build Hub and Spoke Map <-----")
            st.html('<hr style="border: none; height: 3px; background-color: #808080;">')
            
            # Determine view state based on selection
            if submitted:
                if selected_team == 'All Teams':
                    
                    # Show all stadiums
                    view_state = pdk.ViewState(
                        latitude=stadiums_pl['Latitude'].mean(),
                        longitude=stadiums_pl['Longitude'].mean(),
                        zoom=6,
                        pitch=0,
                    )
                    filtered_data = all_players_df
                    display_data = stadiums_pl
                
                else: # user has selected a team to show hub and spoke for
                    origin = stadiums_pl[stadiums_pl['Team'] == selected_team][['Latitude', 'Longitude', 'Name']].iloc[0]
                    origin_dict = {'lat': origin['Latitude'], 'lon': origin['Longitude'], 'name': origin['Name']}
                    destinations = all_players_df[all_players_df['Team'] == selected_team][['latitude', 'longitude', 'name']]
                    destinations_dict = destinations.rename(columns={'latitude': 'lat', 'longitude': 'lon', 'name': 'target_name'}).to_dict('records')

                    st.title(f"Home country of each player on {selected_team} playing at {origin['Name']}")
                    
                    hub.create_hub_spoke_map(origin_dict, destinations_dict)
                    
                    selected_stadium = stadiums_pl[stadiums_pl['Team'] == selected_team].iloc[0]                   
                    image_url = stadiums_pl.loc[stadiums_pl["Team"] == selected_team]['Stadium Image'].iloc[0]
                    badge_url = stadiums_pl.loc[stadiums_pl["Team"] == selected_team]['Badge'].iloc[0]
                    team_url = stadiums_pl.loc[stadiums_pl["Team"] == selected_team]['URL'].iloc[0]
                    stadium_url = stadiums_pl.loc[stadiums_pl["Team"] == selected_team]['Stadium URL'].iloc[0]
                    
                    filtered_data = all_players_df[all_players_df['Team'] == selected_team]

                stadium_config = {
                    "Stadium Image": st.column_config.ImageColumn(),
                    "Badge": st.column_config.ImageColumn(),
                    "URL": st.column_config.LinkColumn(),
                    "Stadium_URL": st.column_config.LinkColumn()
                }
                                   
                st.html('<hr style="border: none; height: 3px; background-color: #808080;">')                
                st.header(f"Players on {selected_team}")
                all_players_df.sort_values(by = "name", inplace = True)
                st.dataframe(all_players_df[all_players_df['Team'] == selected_team][['name', 'flag_path','Badge', 'Player', 'Pos']],
                             column_config={        
                                 "Player": st.column_config.TextColumn(
                                     "Player",
                                     help="Full name of the player"
                                 ),
                                 "name": st.column_config.TextColumn(
                                     "Nationality",
                                     help="Nationality of player"
                                 ),
                                 "Pos": st.column_config.TextColumn(
                                     "Position",
                                     help="Primary playing position (e.g., FW = Forward, MF = Midfielder, DF = Defender, GK = Goalkeeper)"
                                 ),
                                 "flag_path": st.column_config.ImageColumn(
                                     "Flag",
                                     help="Flag of the player's nationality"
                                 ),
                                 "Badge": st.column_config.ImageColumn(
                                     "Badge",
                                     help="Team badge"
                                 )
                             },hide_index=True)  
                
                # create a histogram showing count of players by country
                # # this return sets.  to create a DF to be able to do more
                #sales_summary = transactions.groupby("product_area_name")["sales_cost"].sum().reset_index()      
                
                # Assuming you have a DataFrame called 'players_df' with columns:
                # 'Team', 'Player_Name', 'Country' (or similar)

                
                # Group by country and count players
                country_counts = filtered_data.groupby('name').size().reset_index(name='Player_Count')
                country_counts.rename(columns = {"name" : "Country"}, inplace = True)
                # Sort by player count (descending)
                country_counts = country_counts.sort_values('Player_Count', ascending=False)
                
                # Display the results
                #st.dataframe(country_counts)
                
                st.html('<hr style="border: none; height: 3px; background-color: #808080;">')
                # Optional: Create a bar chart visualization
                st.subheader("Number of Players by Country")
                st.bar_chart(country_counts.set_index('Country')['Player_Count'],
                             x_label = "Countries Represented by Players",
                             y_label = "Number of Players")
                
                # Optional: Show additional statistics
                total_players = len(filtered_data)
                unique_countries = len(country_counts)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Players", total_players)
                with col2:
                    st.metric("Countries Represented", unique_countries)
      
################################################
# Contents of tab6 - Players
################################################

with tab6:
    
    import requests
    st.session_state.active_tab = "Tab 6"
    st.sidebar.empty()
    # Test the new URLs
    # test_urls = all_players_df['flag_path'].head(3).tolist()
    # for url in test_urls:
    #     try:
    #         response = requests.get(url, timeout=5)
    #         st.write(f"✅ {url}: Status {response.status_code}")
    #     except Exception as e:
    #         st.write(f"❌ {url}: Error {e}")
    
    # Create form
    
    with st.form("filter_form"):
        st.write("**Make selections to filter players and click the Filter Data button to see the table.**")
        st.write("**The Team and Position** filter have ***All*** options to see all in that filter.")
        st.write("**To not filter by a stat** simply set the minimum value to 0 for any stat.")
        st.write("**The columns on the table** also allow sorting ascending or descending by clicking on the the 3 dots next to the name.")
        team_filter = st.selectbox("Select Team", ["All"] + list(all_players_df['Team'].unique()))
        position_filter = st.selectbox("Select Position", ["All"] + list(all_players_df['Pos'].unique()))
        country_filter = st.selectbox("Select Country From", ["All"] + list(all_players_df['name'].unique()))
        stat_name = st.selectbox("Select Stat", ['MP', 'Age',  'MP', 'Gls', 'Ast', 'Min', '90s', 'Starts',
               'Subs', 'unSub', 'G+A', 'G-PK', 'PK', 'PKatt', 'PKm'])
        stat_value = st.number_input("Minimum Value", value=0)
                
        submitted = st.form_submit_button("Filter Data")
    
    if submitted:
        # Build list of boolean conditions
        conditions = []
        
        if team_filter != "All":
            conditions.append(all_players_df['Team'] == team_filter)
        
        if position_filter != "All":
            conditions.append(all_players_df['Pos'] == position_filter)
        
        if stat_value > 0:
            conditions.append(all_players_df[stat_name] >= stat_value)
        
        if country_filter != "All":
            conditions.append(all_players_df['name'] == country_filter)
            
        # Combine all conditions with & (AND)
        if conditions:
            final_mask = conditions[0]
            for condition in conditions[1:]:
                final_mask = final_mask & condition
            filtered_df = all_players_df[final_mask]
        else:
            filtered_df = all_players_df
        
        st.dataframe(
            filtered_df,
            column_order=["name", "flag_path", "Player", "Age", 'Badge', 'Team', 'Pos', 'MP', 'Min', 'Perc_Min_Season', '90s', 'Starts',
                  'Subs', 'unSub', 'Gls', 'Ast', 'G+A', 'G-PK', 'PK', 'PKatt', 'PKm'],
            column_config={        
                "Player": st.column_config.TextColumn(
                    "Player",
                    help="Full name of the player"
                ),
                "Age": st.column_config.TextColumn(
                    "Age",
                    help="Current age of the player"
                ),
                "name": st.column_config.TextColumn(
                    "Nationality",
                    help="Nationality of player"
                ),
                "MP": st.column_config.TextColumn(
                    "MP",
                    help="Number of matches the player has played in this season"
                ),
                "Badge": st.column_config.ImageColumn(
                    "Badge", 
                    help="Badge of the current team"
                ),
                "Team": st.column_config.TextColumn(
                    "Team", 
                    help="Current team the player belongs to"
                ),
                "Pos": st.column_config.TextColumn(
                    "Position",
                    help="Primary playing position (e.g., FW = Forward, MF = Midfielder, DF = Defender, GK = Goalkeeper)"
                ),
                "Gls": st.column_config.NumberColumn(
                    "Goals",
                    help="Total goals scored this season"
                ),
                "Ast": st.column_config.NumberColumn(
                    "Assists", 
                    help="Total assists provided this season"
                ),
                "Min": st.column_config.NumberColumn(
                    "Minutes",
                    help="Total minutes played this season"
                ),
                "Perc_Min_Season": st.column_config.ProgressColumn(
                    "% of Season",
                    help="Percentage of total minutes in a season played season"
                ),
                "90s": st.column_config.TextColumn(
                    "90s",
                    help="Minutes played divided by 90 (normalizing to matches played)"
                ),
                "Starts": st.column_config.TextColumn(
                    "Starts",
                    help="Matches started by the player"
                ),
                "Subs": st.column_config.TextColumn(
                    "Subs",
                    help="Matches player did not start so is a substitute"
                ),
                "unSub": st.column_config.TextColumn(
                    "unSub",
                    help="Matches as an unused substitute"
                ),
                "G+A": st.column_config.TextColumn(
                    "G+A",
                    help="Goals and Assists"
                ),
                "G-PK": st.column_config.TextColumn(
                    "G-Pk",
                    help="Non-Penalty goals"
                ),
                "PK": st.column_config.TextColumn(
                    "Pk",
                    help="Penalty kicks made"
                ),
                "PKatt": st.column_config.TextColumn(
                    "PKatt",
                    help="Penalty kicks attempted"
                ),
                "PKm": st.column_config.TextColumn(
                    "PKm",
                    help="Penalty kicks missed"
                ),
                "flag_path": st.column_config.ImageColumn(
                    "Flag",
                    help="Flag of the player's nationality")
            },
            hide_index=True
        )

################################################
# Contents of tab7 - Leader Boards
################################################
        
with tab7:
    
    player_tab, team_tab = st.tabs(["Player", "Team"])
    
    
    with player_tab:
        goals_tab, assists_tab, g_a_tab, age_tab = st.tabs(["Goals", "Assists", "Goals+Assists", "Age"])
        
        with goals_tab:
            
            # Goals leaders
            top_5_players = all_players_df.nlargest(5, "Gls")[["Player", "name", "Gls", "Team"]]
            
            st.write("**Top 5 Players by Goals**")
            st.dataframe(top_5_players, hide_index=True)
            
            ## Bar chart grouped by goals showing count of players
            
            gl_grouping = all_players_df["Gls"].value_counts().reset_index()
            gl_grouping = gl_grouping[gl_grouping["Gls"] != 0]  # Remove rows where Gls is 0
            gl_grouping = gl_grouping.sort_values("count", ascending=False)
            fig = px.bar(gl_grouping, 
                         x="count", 
                         y="Gls", 
                         orientation="h",
                         labels={"count": "Number of Players", "Gls": "Goals"},
                         title="Player Distribution by Goals")
            st.plotly_chart(fig)
         
        with assists_tab:    
            # Assits leaders
            top_5_players = all_players_df.nlargest(5, "Ast")[["Player", "name", "Ast", "Team"]]
            
            st.write("**Top 5 Players by Assists**")
            st.dataframe(top_5_players, hide_index=True)
            
            ast_grouping = all_players_df["Ast"].value_counts().reset_index()
            ast_grouping = ast_grouping[ast_grouping["Ast"] != 0]  # Remove rows where Ast is 0
            ast_grouping = ast_grouping.sort_values("count", ascending=False)
            fig = px.bar(ast_grouping, 
                         x="count", 
                         y="Ast", 
                         orientation="h",
                         labels={"count": "Number of Players", "Ast": "Assits"},
                         title="Player Distribution by Assits")
            st.plotly_chart(fig)
        
        with g_a_tab:
            # G+A leaders
            all_players_ast = all_players_df[all_players_df["Ast"] != 0]  # Remove rows where Ast is 0
            top_5_players = all_players_ast.nlargest(5, "G+A")[["Player", "name", "G+A", "Team"]]
            
            st.write("**Top 5 Players by Goals Plus Assists**")
            st.dataframe(top_5_players, hide_index=True)
        
            gls_ast_grouping = all_players_ast["G+A"].value_counts().reset_index()
            gls_ast_grouping = gls_ast_grouping[gls_ast_grouping["G+A"] != 0]  # Remove rows where Ast is 0
            gls_ast_grouping = gls_ast_grouping.sort_values("count", ascending=False)
            fig = px.bar(gls_ast_grouping, 
                         x="count", 
                         y="G+A", 
                         orientation="h",
                         labels={"count": "Number of Players", "G+A": "Goals Plus Assits"},
                         title="Player Distribution by Goals Plus Assists")
            st.plotly_chart(fig)
        
        with age_tab:
            # Youngest players
            top_5_players = all_players_df.nsmallest(3, "Age")[["Player", "name", "Age", "Team"]]
            
            st.write("**3 Youngest Players in the League**")
            st.dataframe(top_5_players, hide_index=True)
            
            age_grouping = all_players_df["Age"].value_counts().reset_index()
            age_grouping = age_grouping[age_grouping["Age"] != 0]  # Remove rows where Age is 0
            age_grouping = age_grouping.sort_values("Age", ascending=True)
            fig = px.bar(age_grouping, 
                         x="Age", 
                         y="count", 
                         orientation="v",
                         labels={"count": "Number of Players", "Age": "Age of Player"},
                         title="Player Distribution by Age")
            st.plotly_chart(fig)
            
    with team_tab:
        
        gf_tab, ga_tab, gd_tab = st.tabs(["Goals For", "Goals Against", "Goal Diff."])
        
        with gf_tab: 
            st.write("**Team play between 2020-2025**")
            st.write("**Note:** Between these dates Sunderland has only played 3 matches")
            
            # Get total number of matches each team has played
            teams_group_cnt = all_data_df["Team"].value_counts().reset_index()
            teams_group_cnt.rename(columns = {"count" : "MP"}, inplace = True)
            
            # Get total number of GF for each team
            teams_gf_group = all_data_df.groupby("Team")["GF"].sum().reset_index()
            
            # Merge total number of matches (MP) with GF counts
            teams_gf_group = teams_gf_group.merge(teams_group_cnt[["Team", "MP"]], how="left", on="Team")
            
            # calculate goals per match into a new column
            teams_gf_group["Goals Per Match"] = teams_gf_group["GF"] / teams_gf_group["MP"]
            
            # Sort data for display
            teams_gf_group = teams_gf_group.sort_values("Goals Per Match", ascending=False)
            
            # Grab the top 10 teams
            top_10_teams_gf = teams_gf_group.nlargest(10, "Goals Per Match")
            
            # Display the table
            st.write("**Top Teams by Goals For**")
            st.dataframe(top_10_teams_gf, hide_index=True)
            
            # Display the plot
            fig = px.bar(teams_gf_group, 
                         x="Team", 
                         y="Goals Per Match", 
                         orientation="v",
                         labels={"Goals Per Match": "SGoals Per Match", "Team": "Team"},
                         title="Team Distribution by Goals Per Match")
            st.plotly_chart(fig)
            
        with ga_tab:
            st.write("**Team play between 2020-2025**")
            st.write("**Note:** Between these dates Sunderland has only played 3 matches")
            
            # Get total number of matches each team has played
            teams_group_cnt = all_data_df["Team"].value_counts().reset_index()
            teams_group_cnt.rename(columns = {"count" : "MP"}, inplace = True)
            
            # Get total number of GF for each team
            teams_ga_group = all_data_df.groupby("Team")["GA"].sum().reset_index()
            
            # Merge total number of matches (MP) with GF counts
            teams_ga_group = teams_ga_group.merge(teams_group_cnt[["Team", "MP"]], how="left", on="Team")
            
            # calculate goals per match into a new column
            teams_ga_group["Goals Against Per Match"] = teams_ga_group["GA"] / teams_ga_group["MP"]
            
            # Sort data for display
            teams_ga_group = teams_ga_group.sort_values("Goals Against Per Match", ascending=False)
            
            # Grab the top 10 teams
            top_10_teams_ga = teams_ga_group.nlargest(10, "Goals Against Per Match")
            
            # Display the table
            st.write("**Top Teams by Goals For**")
            st.dataframe(top_10_teams_ga, hide_index=True)
        
            
            fig = px.bar(teams_ga_group, 
                         x="Team", 
                         y="Goals Against Per Match", 
                         orientation="v",
                         labels={"Goals Against Per Match": "Goals Against Per Match", "Team": "Team"},
                         title="Team Distribution by GA per Match")
            st.plotly_chart(fig)
            
        with gd_tab:
            st.write("**Team play between 2020-2025**")
            st.write("**Note:** Between these dates Sunderland has only played 3 matches")
            
            teams_gd_group = all_data_df.groupby("Team")["GD"].sum().reset_index()
            teams_gd_group = teams_gd_group.sort_values("GD", ascending=False)
            top_10_teams_gd = teams_gd_group.nlargest(10, "GD")
            
            st.write("**Top Teams by Goals For**")
            st.dataframe(top_10_teams_gd, hide_index=True)
            
            fig = px.bar(teams_gd_group, 
                         x="Team", 
                         y="GD",
                         color="GD",
                         color_continuous_scale=['red', 'lightgray', 'green'],
                         color_continuous_midpoint=0,
                         orientation="v",
                         labels={"GD": "Sum of Goal Diff.", "Team": "Team"},
                         title="Team Distribution by GD")
            st.plotly_chart(fig)
        

        
        
        
        
        
        
        
        
        
        
        