# import libraries

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import pickle
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Bayesian calculation class and function

class BayesianTeamAnalyzer:
    """
    Bayesian analysis for comparing teams with different sample sizes.
    Handles the problem where teams with fewer matches appear artificially superior.
    """
    
    def __init__(self, stat_names=None):
        self.stat_names = stat_names or ['SoT', 'GF', 'Poss', 'Long_Completes', 'Success', 'Blocks']
        self.results = {}
    
    def empirical_bayes_shrinkage(self, team_data, prior_mean, prior_precision=1.0):
        """
        Apply empirical Bayes shrinkage to team statistics.
        
        Parameters:
        - team_data: array of match-level statistics for the team
        - prior_mean: league average (or dominant team's average as prior)
        - prior_precision: confidence in the prior (higher = more shrinkage)
        """
        n = len(team_data)
        sample_mean = np.mean(team_data)
        sample_var = np.var(team_data, ddof=1) if n > 1 else 1.0
        
        # Empirical Bayes shrinkage formula
        # Shrinkage intensity depends on sample size and variance
        tau = prior_precision  # Prior precision
        likelihood_precision = n / sample_var if sample_var > 0 else n
        
        # Posterior mean (shrunk estimate)
        posterior_precision = tau + likelihood_precision
        shrunk_mean = (tau * prior_mean + likelihood_precision * sample_mean) / posterior_precision
        
        # Calculate shrinkage factor for interpretation
        shrinkage_factor = tau / posterior_precision
        
        return {
            'raw_mean': sample_mean,
            'shrunk_mean': shrunk_mean,
            'shrinkage_factor': shrinkage_factor,
            'posterior_precision': posterior_precision,
            'n_matches': n
        }
    
    def hierarchical_bayes_estimate(self, all_teams_data, target_team_data):
        """
        Hierarchical Bayesian model using all teams to inform priors.
        
        Parameters:
        - all_teams_data: list of arrays, each containing one team's data
        - target_team_data: array of target team's data to estimate
        """
        # Estimate hyperpriors from all teams
        all_means = [np.mean(team) for team in all_teams_data if len(team) > 0]
        
        # Population-level parameters
        mu_0 = np.mean(all_means)  # Population mean
        tau_0 = 1 / np.var(all_means) if len(all_means) > 1 else 1.0  # Population precision
        
        # Apply shrinkage using population parameters as prior
        return self.empirical_bayes_shrinkage(target_team_data, mu_0, tau_0)
    
    def compare_teams_bayesian(self, team1_data, team2_data, stat_names=None):
        """
        Compare two teams using Bayesian analysis.
        Handles different sample sizes appropriately.
        """
        if stat_names is None:
            stat_names = self.stat_names
            
        results = {}
        
        for i, stat in enumerate(stat_names):
            # Extract statistic for both teams
            team1_stat = [match[i] for match in team1_data]
            team2_stat = [match[i] for match in team2_data]
            
            # Use Team 1 as prior (since it has more data)
            team1_mean = np.mean(team1_stat)
            
            # Apply Bayesian shrinkage to both teams
            team1_result = self.empirical_bayes_shrinkage(team1_stat, team1_mean, prior_precision=0.1)
            team2_result = self.empirical_bayes_shrinkage(team2_stat, team1_mean, prior_precision=1.0)
            
            results[stat] = {
                'team1': team1_result,
                'team2': team2_result,
                'difference_raw': team2_result['raw_mean'] - team1_result['raw_mean'],
                'difference_shrunk': team2_result['shrunk_mean'] - team1_result['shrunk_mean']
            }
        
        self.results = results
        return results
    
    def credible_intervals(self, team_data, prior_mean, prior_precision=1.0, confidence=0.95):
        """
        Calculate Bayesian credible intervals (analogous to confidence intervals).
        """
        n = len(team_data)
        sample_mean = np.mean(team_data)
        sample_var = np.var(team_data, ddof=1) if n > 1 else 1.0
        
        # Posterior parameters
        tau = prior_precision
        likelihood_precision = n / sample_var if sample_var > 0 else n
        posterior_precision = tau + likelihood_precision
        posterior_mean = (tau * prior_mean + likelihood_precision * sample_mean) / posterior_precision
        posterior_var = 1 / posterior_precision
        
        # Credible interval
        alpha = 1 - confidence
        z = stats.norm.ppf(1 - alpha/2)
        margin = z * np.sqrt(posterior_var)
        
        return {
            'mean': posterior_mean,
            'lower': posterior_mean - margin,
            'upper': posterior_mean + margin,
            'margin': margin
        }
    
    
    def summary_table(self):
        """
        Create a summary table of the Bayesian analysis.
        """
        if not self.results:
            print("No results available. Run compare_teams_bayesian first.")
            return None
        
        summary_data = []
        team1_baye_list = []
        team2_baye_list = []

        for stat, data in self.results.items():
            summary_data.append({
                'Statistic': stat,
                'Team1_Raw': f"{data['team1']['raw_mean']:.2f}",
                'Team1_Bayesian': f"{data['team1']['shrunk_mean']:.2f}",
                'Team2_Raw': f"{data['team2']['raw_mean']:.2f}",
                'Team2_Bayesian': f"{data['team2']['shrunk_mean']:.2f}",
                'Raw_Difference': f"{data['difference_raw']:.2f}",
                'Bayesian_Difference': f"{data['difference_shrunk']:.2f}",
                'Team2_Shrinkage': f"{data['team2']['shrinkage_factor']:.2f}"
            })
            team1_baye_list.append(data['team1']['shrunk_mean'])
            team2_baye_list.append(data['team2']['shrunk_mean'])
        
        df = pd.DataFrame(summary_data)
        team1_baye_df = pd.DataFrame([team1_baye_list], columns=["SoT", "GF", "Poss", "Long_Cmp", "Succ", "Blocks"])
        team2_baye_df = pd.DataFrame([team2_baye_list], columns=["SoT", "GF", "Poss", "Long_Cmp", "Succ", "Blocks"])
        
        return df, team1_baye_df, team2_baye_df

def highlight_multiple_conditions(row):
    """Apply different colors based on prediction being correct or not"""
    if row['Result'] == row['Predicted']:
        return ['background-color: #ccffcc'] * len(row)  # Green
    elif row['Result'] != row['Predicted']:
        return ['background-color: #ffcccc'] * len(row)  # Red
    else:
        return [''] * len(row)

# 24-25 seasons
#clf_reduced = pickle.load(open("web_app_Predictor.p", "rb"))
#clf_reduced_name = 'web_app_Predictor_24_25'
#data_for_avg = pickle.load(open("web_app_data.p", "rb"))

# 20-26 seaons
clf_reduced = joblib.load('premier_random_forest_20_26_prediction.joblib')
clf_reduced_name = 'premier_random_forest_20_26'
data_for_avg = joblib.load('premier_random_forest_20_26_prediction_data.joblib')

#All matches with results only

win_count_df = data_for_avg.groupby("Team")["Result"].value_counts().reset_index()

# All matches with data, team, opp, and result
all_data_df = pd.read_csv("match_data_20_26.csv")
all_data_df['Result'] = all_data_df['Result'].str.split(' ').str[0]

columns_to_keep = ['Date','Team', 'Opp', 'Result']
match_result_lookup = all_data_df[columns_to_keep]

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
    st.text("Date Range: 9/12/2020 and 8/31/2025.")
    
    # Add high level instructions
    st.subheader("Select the Home and Away teams, along with their team weights to boost the probability. Click submit and scroll down to see the likelihood and pick of the match result")
    st.text("Note: Team weights should range beteen 0 and 2.")
    
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
         "Newcastle Utd", "Nott'ham Forest", 
         "Sunderland", "Tottenham", 
         "West Ham", "Wolves"
         ), width=150, key="team1_name"
    )
    
    team1_weight = left.number_input(
        label = "Home team weight",
        min_value = 0.00,
        max_value = 2.00,
        value = 1.00,
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
         "Newcastle Utd", "Nott'ham Forest", 
         "Sunderland", "Tottenham", 
         "West Ham", "Wolves"
         ), width=150, key="team2_name"
    )
       
    team2_weight = left.number_input(
        label = "Away team weight",
        min_value = 0.00,
        max_value = 2.00,
        value = 1.00,
        width=150, key = "team2_weight")

    # Process prediction button
       
    if left.button("Process", icon=":material/online_prediction:"):
        
        # For analysis only.
        # team1_name="Arsenal"
        # team2_name="Leeds United"
        # teams = data_for_avg['Team']
        
        # Remove results feature, not needed for creating averages
        data_for_avg.drop(['Result'], axis=1, inplace = True)
        
        # First collect all data for each team in to separate dataframes.
        tmp_team1 = data_for_avg[data_for_avg['Team'] == team1_name].copy()
        tmp_team1.drop(['Team'], axis=1, inplace = True)
    
        tmp_team2 = data_for_avg[data_for_avg['Team'] == team2_name].copy()
        tmp_team2.drop(['Team'], axis=1, inplace = True)
        len(tmp_team1)
        len(tmp_team2)
        
        #print(abs(len(tmp_team1) - len(tmp_team2))

        # if one team has played less matches than the other apply a Baysian Shrinkage
        # to make sure the averages for that team are reliable.  This is more pronounced
        # for the teams that are newer to the premier league.
        
        if (abs(len(tmp_team1) - len(tmp_team2)) > 0):
            bay_application = 'Applying Bayesien Shrinkage due to the imbalance of the number in matches of each team in historic data'
            
            # Convert DataFrames to list of lists format expected by Bayesian analyzer
            team1_data_list = tmp_team1.values.tolist()  # Convert DataFrame to list of lists
            team2_data_list = tmp_team2.values.tolist()  # Convert DataFrame to list of lists
            
            # Initialize analyzer
            analyzer = BayesianTeamAnalyzer()
            
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

        
        # When Win probabilitie are extremely close, fovor the team with the most wins historically.
        # if (((away_win - home_win) ** 2) * 100) < 0.041:
        #     team_wins = win_count_df.loc[(win_count_df["Team"].isin(["Manchester Utd", "Arsenal"])) & (win_count_df["Result"] == "W"),["Team","count"]].reset_index()        
        #     max_row_index = team_wins['count'].idxmax()
        #     team_name_max_wins = team_wins.loc[max_row_index, 'Team']
            
        #     if (team_name_max_wins == team1_name):
        #         team1_weight = 2.0
        #     elif (team_name_max_wins == team2_name):
        #         team2_weight = 2.0
        
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
        
        middle.markdown(f"""
            <span style="color: red;">Our Prediction Using: {clf_reduced_name}</span><br><br>
            <span style="color: red;">Home Win: {home_win_prob}</span><br><br>
            <span style="color: red;">Tie: {tie_prob}</span><br><br>
            <span style="color: red;">Away Win: {away_win_prob}</span><br><br>
            <span style="color: red;">Pick: {pick}</span><br><br>
            <span style="color: red;">Statistical Logic: {bay_application}</span><br><br>
            <span style="color: red;">Actual Probabilities: {predict_proba}</span><br><br>
            <span style="color: red;">Favoring: {favoring} - Home Win: {home_win} Away Win: {away_win}</span>
            """, unsafe_allow_html=True)   
        
        middle.dataframe(combined_avg)
            
        bay_application = ''
        favoring = ''
    
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
                      [6,"Arsenal", "Leeds United", 1, 1],
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
    game_week4_weighted_list = [[1,"Arsenal", "Nott'm Forest", 1, 1],
                      [2,"Bournemouth", "Brighton", 1, 1],
                      [3,"Crystal Palace", "Sunderland", 1, 1],
                      [4,"Everton", "Aston Villa",1, 1],
                      [5,"Fulham", "Leeds United",1 , 1],
                      [6,"Newcastle", "Wolves" ,1, 1],
                      [7,"West Ham", "Spurs", 1, 1], 
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
    gw_1_actuals_list = [[1,"Liverpool", "Bournemouth", "4-2", "Home Win", "Home Win"],
                      [2,"Aston Villa", "Newcastle", "0-0", "Tie", "Home Win"],
                      [3,"Brighton", "Fulham", "1-1", "Tie", "Tie"],
                      [4,"Sunderland", "West Ham", "3-0", "Home Win", "Home Win"],
                      [5,"Spurs", "Burnley", "3-0", "Home Win", "Home Win"],
                      [6,"Wolves", "Man City", "0-4", "Away Win", "Away Win"],
                      [7,"Nott'm Forest", "Brentford", "3-1", "Home Win", "Tie"], 
                      [8,"Chelsea", "Crystal Palace", "0-0", "Tie", "Home Win"],
                      [9,"Man Utd", "Arsenal", "0-1", "Away Win", "Away Win"],
                      [10,"Leeds United", "Everton", "1-0", "Home Win", "Tie"]]
    gw_1_actuals = pd.DataFrame(gw_1_actuals_list, columns=["match ID","Home","Away", "Score", "Result", "Predicted"])

    # Game week 2    
    gw_2_actuals_list = [[1,"West Ham", "Chelsea", "1-5", "Away Win", "Away Win"],
                      [2,"Man City", "Spurs", "0-2", "Away Win", "Home Win"],
                      [3,"Bournemouth", "Wolves", "1-0", "Home Win", "Home Win"],
                      [4,"Brentford", "Aston Villa", "1-0", "Home Win", "Away Win"],
                      [5,"Burnley", "Sunderland", "2-0", "Home Win", "Home Win"],
                      [6,"Arsenal", "Leeds United", "5-0", "Home Win", "Home Win"],
                      [7,"Crystal Palace", "Nott'm Forest", "1-1", "Tie", "Tie"], 
                      [8,"Everton", "Brighton", "2-0", "Home Win", "Home Win"],
                      [9,"Fulham", "Man Utd", "1-1", "Tie", "Away Win"],
                      [10,"Newcastle", "Liverpool", "2-3", "Away Win", "Away Win"]]
    gw_2_actuals = pd.DataFrame(gw_2_actuals_list, columns=["match ID","Home","Away", "Score", "Result", "Predicted"])

    # game week 3
    gw_3_actuals_list = [[1,"Chelsea", "Fulham", "2-0", "Home Win", "Home Win"],
                      [2, "Man Utd", "Burnley", "3-2", "Home Win", "Home Win"],
                      [3, "Sunderland", "Brentford", "2-1", "Home Win", "Home Win"],
                      [4, "Spurs", "Bournemouth", "0-1", "Away Win", "Home Win"],
                      [5, "Wolves", "Everton", "2-3", "Away Win", "Home Win"],
                      [6, "Leeds United", "Newcastle", "0-0", "Tie", "Away Win"],
                      [7, "Brighton", "Man City", "2-1", "Home Win", "Away Win"], 
                      [8, "Nott'm Forest", "West Ham", "0-3", "Away Win", "Tie"],
                      [9, "Liverpool", "Arsenal", "1-0", "Home Win", "Home Win"],
                      [10,"Aston Villa", "Crystal Palace", "0-3", "Away Win", "Home Win"]]
    gw_3_actuals = pd.DataFrame(gw_3_actuals_list, columns=["match ID","Home","Away", "Score", "Result", "Predicted"])

    # game week 4
    gw_4_actuals_list = [[1,"Arsenal", "Nott'm Forest", "3-0", "Home Win", "Home Win"],
                    [2,"Bournemouth", "Brighton", "2-1", "Home Win", "Away Win"],
                    [3,"Crystal Palace", "Sunderland", "0-0", "Tie", "Tie"],
                    [4,"Everton", "Aston Villa", "0-0", "Tie", "Away Win"],
                    [5,"Fulham", "Leeds United", "1-0", "Home Win", "Home Win"],
                    [6,"Newcastle", "Wolves", "1-0", "Home Win", "Home Win"],
                    [7,"West Ham", "Spurs", "0-3", "Away Win", "Away Win"],
                    [8,"Brentford", "Chelsea", "2-2", "Tie", "Away Win"],
                    [9,"Burnley", "Liverpool", np.nan, np.nan, "Away Win"],
                    [10,"Man City", "Man Utd", np.nan, np.nan, "Home Win"]]
    gw_4_actuals = pd.DataFrame(gw_4_actuals_list, columns=["match ID","Home","Away", "Score", "Result", "Predicted"])
    
    # mapping of game selection text to the correct dataframe
    actuals_week_mapping = {
        "game_week1": gw_1_actuals,
        "game_week2": gw_2_actuals,
        "game_week3": gw_3_actuals,
        "game_week4": gw_4_actuals
    }
    
    # Display Actual information
    
    middle.subheader("Actual VS. Predicted predicted by Game Week")
    
    gw_num_actuals = middle.selectbox(
        "Pick a game week:",
        ("game_week1",
         "game_week2",
         "game_week3",
         "game_week4"
         ),  key="gw_num_actuls")
    
    # Get the selected DataFrame and display it
    selected_dataframe = actuals_week_mapping.get(gw_num_actuals)
    if selected_dataframe is not None:
        if  selected_dataframe.isna().sum().sum() > 0:
            predict_calc_df = selected_dataframe.dropna(how = "any" )
        else:
            predict_calc_df = selected_dataframe
        num_right =len(predict_calc_df.loc[(predict_calc_df["Predicted"] == predict_calc_df["Result"])])
        total = len(predict_calc_df)
        accuracy = num_right/total
        middle.text(f"Predicted correct accuracy: {num_right} of {total} played - {accuracy:.0%}")
        styled_df = selected_dataframe.style.apply(highlight_multiple_conditions, axis=1)        
        middle.dataframe(styled_df, column_order= ("Home","Away", "Score", "Predicted"), hide_index=True)

    else:
        middle.write(f"DataFrame for {gw_num_actuals} not yet implemented")
        middle.text("Predicted correct accuracy: NONE")

    #############
    # Right column
    #############
    
    # Import table CSV file with all tables in it
    table_all_df = pd.read_csv("tables_all.csv")
    
    # create a dataframe for each game week from table_all_df and selecting  on gw_num 
    table_1_game_df = table_all_df[table_all_df["gw_num"] == 1]
    table_2_game_df = table_all_df[table_all_df["gw_num"] == 2]
    table_3_game_df = table_all_df[table_all_df["gw_num"] == 3] 
    table_4_game_df = table_all_df[table_all_df["gw_num"] == 4] 
    
    # Mapping for selecte gameweek to correct table dataframe
    table_mapping = {
        "post game week 1": table_1_game_df,
        "post game week 2": table_2_game_df,
        "post game week 3": table_3_game_df,
        "post game week 4": table_4_game_df,
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
         "post game week 20"),  key="full_tables")
    
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
    # Import table CSV file with all tables in it
    table_all_df = pd.read_csv("tables_all.csv")
    
    # create a dataframe for each game week from table_all_df and selecting  on gw_num 
    table_1_game_df = table_all_df[table_all_df["gw_num"] == 1]
    table_2_game_df = table_all_df[table_all_df["gw_num"] == 2]
    table_3_game_df = table_all_df[table_all_df["gw_num"] == 3] 
    table_4_game_df = table_all_df[table_all_df["gw_num"] == 4]
    
    # Mapping of selected text to proper dataframe
    table_mapping = {
        "post game week 1": table_1_game_df,
        "post game week 2": table_2_game_df,
        "post game week 3": table_3_game_df,
        "post game week 4": table_4_game_df,
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
        st.dataframe(selected_dataframe, column_order = ("Pos","Team","Pl","W","D","L","GF","GA","GD","Pts"), hide_index=True)

    else:
        middle.write(f"DataFrame for {gw_num_tables} not yet implemented")
    
        
    st.subheader("Matches played by each team in the historic dataset")    

    games_played_df = data_for_avg['Team'].value_counts().reset_index()
    st.dataframe(games_played_df, hide_index=True)
    
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
    
    
    
    
    
