###################
# import libraries
###################
import streamlit as st
import time
with st.spinner("Wait for it...", show_time=True):
   
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
    
    # Local libraries (not required to listed in requirements.txt on github)
    import Analytics.Advanced_Analytics as aa
    import Plot.timeseries_plots as tp
    import Plot.performance_plots as pp
    import Plot.table_style as ts
    import Data.preprocess as preproc
    import Stats.bayesian_analysis as ba
    import Map_code.hub_spoke_map as hub
    import Data.stadiums_merge as sm
    import Plot.form_table
    import Data.create_actuals as create_actuals

    ########################
    # Import model and data
    ########################
    
    # set pandas display option
    pd.set_option('display.max_columns', None)
    
    ###############################################################################
    #Joblib files were created from premier_league_new_data_processing.py in the main
    #app directory of Premier_prediction.
    #
    #01-Oct-2025: Added data from Sept 2025
    ###############################################################################
    
    # For more increased performance and to reduce massice data processing on every
    # user interaction, create these funtions to use streamlit cacheing.
    # use @st.cache_resource for: ML models, database connections, global objects
    # use st.cache_data for: DataFrames, lists, processed data
    # Don't cache: User inputs, session state, dynamic calculations based on user selections
    
    @st.cache_resource
    def load_model():
        return joblib.load('Models/premier_random_forest_2025_20260108_prediction.joblib')
    
    @st.cache_data
    def load_match_data():
        all_data_df = pd.read_csv("Data/match_data_2025_20260108.csv")
        all_data_df['Result'] = all_data_df['Result'].str.split(' ').str[0]
        all_data_df["match_counter"] = 1
        return all_data_df
    
    @st.cache_data
    def load_avg_data():
        data_for_avg = joblib.load('Data/premier_random_forest_2025_20260108_prediction_data.joblib')
        return data_for_avg 
    
    @st.cache_data
    def create_player_data(stadiums_pl, countries_df):
        
        # Load
        all_players_df = pd.read_csv("Data/players_25_26.csv")
        
        # clean up
        all_players_df.drop(["Season", "Comp", "-9999"], axis=1, inplace=True)
        
        #country_codes_df = pd.read_csv("Data/world.csv")

        all_players_df['Nation'] = all_players_df['Nation'].str.replace(r'[A-Z]{3}', '', regex=True).str.strip()
        
        all_players_df['Nation'] = all_players_df['Nation'].replace({'eng': 'gb',
                                                                     'wls': 'gb-wls',
                                                                     'nir': 'gb-nir',
                                                                     'sct': 'gb-sct'})
        
        # Add flag_path URLs
        all_players_df['flag_path'] = all_players_df['Nation'].apply(
            lambda x: f"https://flagcdn.com/64x48/{x}.png"
        )
        
        all_players_df['Nation'] = all_players_df['Nation'].str.upper()
        
        # Add extra stadium data
        #stadiums_pl = sm.stadium_merge(stadiums_pl)
        
        # Grab badge from stadiums-pl and add to players
        all_players_df['Team'] = all_players_df['Team'].replace({"Nott'ham Forest": 'Nottingham Forest'})
        all_players_df = all_players_df.merge(stadiums_pl[["Team", "Badge"]], how="left", on="Team")
        
        # Add in a % of total minutes for the season 28*90
        all_players_df["Perc_Min_Season"] = all_players_df["Min"] / (28 * 90)
        all_players_df.fillna(value = 0, inplace=True)    
        
        # Import country location data to be able to build hub and spoke maps for teams and players.
        # countries_df = pd.read_csv("Data/countries.csv")
        
        all_players_df = all_players_df.merge(countries_df[['country', 'latitude', 'longitude', 'name']], how="left", left_on='Nation', right_on='country')
        
        all_players_df.isna().sum()
        return all_players_df
    
    @st.cache_data
    def create_team_stats(all_data_df):
        # add win % to add_data_df.  First add a match counter set to 1 for all rows.  Used to count up all matches for a team.
        #all_data_df["match_counter"] = 1
        
        # create summary table with counts
        team_stats = all_data_df.groupby('Team').agg(
            total_matches=('match_counter', 'sum'),
            wins=('Result', lambda x: (x == 'W').sum()),
            draws=('Result', lambda x: (x == 'D').sum()),
            losses=('Result', lambda x: (x == 'L').sum())
        ).reset_index()
        
        #create win_rate in summary table
        team_stats['win_rate'] = team_stats['wins'] / team_stats['total_matches']
        return team_stats
    
    @st.cache_data
    def create_timeseries_data(all_data_df):
        all_unique = all_data_df[all_data_df['Team'] < all_data_df['Opp']]
        analysis_df = all_unique
        
        # Timeseries data for the data info tab
        timeseries_df = analysis_df.groupby(["Date", "Result"])["Result"].value_counts().reset_index()
        timeseries_df['Date'] = pd.to_datetime(timeseries_df['Date'])
        
        # Dealing with dates
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
        
        return timeseries_df
    
    @st.cache_data
    def create_merge_data(data_for_avg):
        merged_df = pd.merge(data_for_avg, enhanced_results_summary_df, how = "inner", on = "Team")
        
        return merged_df
    
    @st.cache_data
    def create_accuracy_data():
        accuracy_tracking = pd.DataFrame({"Game Week" : ["GW 01", "GW 02", "GW 03", "GW 04", "GW 05", 
                                                         "GW 06", "GW 07", "GW 08", "GW 09", "GW 10", 
                                                         "GW 11", "GW 12", "GW 13", "GW 14", "GW 15",
                                                         "GW 16", "GW 17", "GW 18", "GW 19", "GW 20",
                                                         "GW 21"],
                                          
                                          "Accuracy" : [60, 70, 40, 70, 50, 
                                                        40, 60, 40, 40, 50, 
                                                        50, 30, 50, 30, 50,
                                                        50, 60, 50, 30, 40,
                                                        40],
                                          
                                          "Mean" : [60.0, 65.0, 56.7, 60.0, 58.0, 
                                                    55.0, 55.7, 53.8, 52.2, 52.0, 
                                                    51.8, 50.0, 50.0, 48.6, 48.7,
                                                    48.8, 49.4, 49.4, 48.4, 48.0,
                                                    47.6],
                                          
                                          "Mean of Mean" : [58.2, 58.2, 58.2, 58.2, 58.2,
                                                            58.2, 58.2, 58.0, 57.4, 57.4, 
                                                            56.4, 55.8, 53.4, 54.9, 54.5,
                                                            54.1, 53.9, 53.6, 53.3, 53.1,
                                                            52.8]})
        
        
        acc = [60, 70, 40, 70, 50, 
               40, 60, 40, 40, 50, 
               50, 30, 50, 30, 50,
               50, 60, 50, 30, 40, 
               40]
        
        acc_mean = sum(acc) / len(acc)
        #print(acc_mean)
        
        mean_mean = sum([60.0, 65.0, 56.7, 60.0, 58.0, 
                         55.0, 55.7, 53.8, 52.2, 52.0, 
                         51.8, 50.0, 50.0, 48.6, 48.7,
                         48.8, 49.4, 49.4, 48.4, 48.0,
                         47.6]) / len([60.0, 65.0, 56.7, 60.0, 58.0, 
                                                         55.0, 55.7, 53.8, 52.2, 52.0, 
                                                         51.8, 50.0, 50.0, 48.6, 48.7,
                                                         48.8, 49.4, 49.4, 48.4, 48.0,
                                                         47.6])
        #print(mean_mean)
                                                         
        return accuracy_tracking, mean_mean
    
    @st.cache_data
    def create_stadium_data(all_data_df):
        
        import Data.stadiums_merge as sm
        
        stadium_data = pd.read_csv("Data/stadiums.csv")
        
        all_df = pd.DataFrame(all_data_df["Team"].unique(), columns = ["Team"])
        idx = all_df[all_df['Team'] == "Nott'ham Forest"].index
        all_df.loc[idx, 'Team'] = "Nottingham Forest"
        idx = all_df[all_df['Team'] == "Newcastle Utd"].index
        all_df.loc[idx, 'Team'] = "Newcastle United"
        
        all_df_for_Stadium_merge = all_df.copy()
        
        stadiums_pl = pd.merge(all_df_for_Stadium_merge, stadium_data, how="left", on="Team")
        
        # Add extra stadium data
        stadiums_pl = sm.stadium_merge(stadiums_pl)
        
        return stadiums_pl
    
    @st.cache_data
    def create_country_data():
        countries_df = pd.read_csv("Data/countries.csv")
        
        return countries_df
    
    @st.cache_data
    def create_country_code_data():
        country_codes_df = pd.read_csv("Data/world.csv")
        
        return country_codes_df
    
    @st.cache_data
    def create_attendance_date(stadiums_pl):
        attendance_df = pd.read_csv("Data/attendance_tracking_20250931.csv")
        
        # Merge in stadium data with attendance
        attendance_df = attendance_df.merge(stadiums_pl, how="left", on="Team")
        
        return attendance_df
    
    @st.cache_data
    def create_coach_data(countries_df, stadiums_pl):
        # Load coach data
        coaches_df = pd.read_csv("Data/coaches.csv")
        #coaches_df.isna().sum()
        coaches_df.drop([" Ref"], axis= 1, inplace=True)
        
        # Deal with dates
        # Convert Start and End to datetime
        coaches_df['Start'] = pd.to_datetime(coaches_df['Start'], format='%d %B %Y', errors='coerce')
        coaches_df['End'] = pd.to_datetime(coaches_df['End'], format='%d %B %Y', errors='coerce')
        
        # Deal with NaT in date columns
        # First, ensure Duration is numeric (in days)
        coaches_df['Duration'] = pd.to_numeric(coaches_df['Duration'], errors='coerce')
        
        # Fill missing End dates using Start + Duration
        mask_missing_end = coaches_df['End'].isna() & coaches_df['Start'].notna()
        coaches_df.loc[mask_missing_end, 'End'] = (
            coaches_df.loc[mask_missing_end, 'Start'] + pd.to_timedelta(coaches_df.loc[mask_missing_end, 'Duration'], unit='D')
        )
        
        # Fill missing Start dates using End - Duration
        mask_missing_start = coaches_df['Start'].isna() & coaches_df['End'].notna()
        coaches_df.loc[mask_missing_start, 'Start'] = (
            coaches_df.loc[mask_missing_start, 'End'] - pd.to_timedelta(coaches_df.loc[mask_missing_start, 'Duration'], unit='D')
        )
        
        # Add national flag data.
     
        # first convert Nationality to 2 char value
        coaches_df = coaches_df.merge(countries_df[['country', 'latitude', 'longitude', 'name']], how="left", left_on='Nationality', right_on='name')
    
        # convert 2 char code to lower case to be able to create the correct link for the flag image.
        coaches_df['country'] = coaches_df['country'].str.lower()
        
        # Add flag_path URLs
        coaches_df['flag_path'] = coaches_df['country'].apply(
            lambda x: f"https://flagcdn.com/64x48/{x}.png"
        )
        
        # Correct flags for UK countries
        idx = coaches_df[coaches_df['name'] == "Scotland"].index
        coaches_df.loc[idx, 'flag_path'] = "https://upload.wikimedia.org/wikipedia/commons/1/10/Flag_of_Scotland.svg"
        
        idx = coaches_df[coaches_df['name'] == "Wales"].index
        coaches_df.loc[idx, 'flag_path'] = "https://upload.wikimedia.org/wikipedia/commons/d/dc/Flag_of_Wales.svg"
        
        idx = coaches_df[coaches_df['name'] == "Northern Ireland"].index
        coaches_df.loc[idx, 'flag_path'] = "https://upload.wikimedia.org/wikipedia/commons/4/45/Flag_of_Ireland.svg"
        
        idx = coaches_df[coaches_df['name'] == "Republic of Ireland"].index
        coaches_df.loc[idx, 'flag_path'] = "https://upload.wikimedia.org/wikipedia/commons/4/45/Flag_of_Ireland.svg"
    
        
        # grab team badge image where available.
        coaches_df = coaches_df.merge(stadiums_pl[["Team", "Badge"]], how="left", on="Team")
        
        # Remove symbols from names
        coaches_df['Name'] = coaches_df['Name'].str.replace('‡', '').str.replace('†', '').str.replace('§', '').str.strip()
       
        return coaches_df

    
    @st.cache_data
    def create_table_all():
        table_all_df = pd.read_csv("Data/tables_all.csv")
        
        return table_all_df
    
    @st.cache_data
    def create_table_data():
 
        # create a dataframe for each game week from table_all_df and selecting  on gw_num 
        table_all_df = create_table_all()
        
        table_1_game_df = table_all_df[table_all_df["Pl"] == 1]
        table_2_game_df = table_all_df[table_all_df["Pl"] == 2]
        table_3_game_df = table_all_df[table_all_df["Pl"] == 3] 
        table_4_game_df = table_all_df[table_all_df["Pl"] == 4] 
        table_5_game_df = table_all_df[table_all_df["Pl"] == 5]
        table_6_game_df = table_all_df[table_all_df["Pl"] == 6]    
        table_7_game_df = table_all_df[table_all_df["Pl"] == 7]
        table_8_game_df = table_all_df[table_all_df["Pl"] == 8]
        table_9_game_df = table_all_df[table_all_df["Pl"] == 9]
        table_10_game_df = table_all_df[table_all_df["Pl"] == 10]
        table_11_game_df = table_all_df[table_all_df["Pl"] == 11]
        table_12_game_df = table_all_df[table_all_df["Pl"] == 12]
        table_13_game_df = table_all_df[table_all_df["Pl"] == 13]
        table_14_game_df = table_all_df[table_all_df["Pl"] == 14]
        table_15_game_df = table_all_df[table_all_df["Pl"] == 15]
        table_16_game_df = table_all_df[table_all_df["Pl"] == 16]
        table_17_game_df = table_all_df[table_all_df["Pl"] == 17]
        table_18_game_df = table_all_df[table_all_df["Pl"] == 18]
        table_19_game_df = table_all_df[table_all_df["Pl"] == 19]
        table_20_game_df = table_all_df[table_all_df["Pl"] == 20]
        table_21_game_df = table_all_df[table_all_df["Pl"] == 21]

        
        #return table_1_game_df, table_2_game_df, table_3_game_df, table_4_game_df, table_5_game_df, table_6_game_df, table_7_game_df, table_8_game_df
        return table_21_game_df
        
    # Then in main code:
    clf_reduced = load_model()
    all_data_df = load_match_data()
    data_for_avg = load_avg_data()
    
    
    
    #clf_reduced = joblib.load('Models/premier_random_forest_2020_20250931_prediction.joblib')
    clf_reduced_name = 'premier_random_forest_2025_20260108'
    #data_for_avg = joblib.load('Data/premier_random_forest_2020_20250931_prediction_data.joblib')
    
    ########################
    # Import match data
    ########################
   
    # All matches with data, team, opp, and result
    #all_data_df = pd.read_csv("Data/match_data_2020_20250931.csv")
    #all_data_df['Result'] = all_data_df['Result'].str.split(' ').str[0]
    # all_unique = all_data_df[all_data_df['Team'] < all_data_df['Opp']]
    # analysis_df = all_unique
    
    # add win % to add_data_df.  First add a match counter set to 1 for all rows.  Used to count up all matches for a team.
    #all_data_df["match_counter"] = 1

    # create summary table with counts
    # team_stats = all_data_df.groupby('Team').agg(
    #     total_matches=('match_counter', 'sum'),
    #     wins=('Result', lambda x: (x == 'W').sum()),
    #     draws=('Result', lambda x: (x == 'D').sum()),
    #     losses=('Result', lambda x: (x == 'L').sum())
    # ).reset_index()
    
    # #create win_rate in summary table
    # team_stats['win_rate'] = team_stats['wins'] / team_stats['total_matches']
    
    team_stats = create_team_stats(all_data_df)
    
    ########################
    # Create timeseries data.  
    ########################
    
    #timeseries_df = create_timeseries_data(all_data_df)
    
    # Timeseries data for the data info tab
    # timeseries_df = analysis_df.groupby(["Date", "Result"])["Result"].value_counts().reset_index()
    # timeseries_df['Date'] = pd.to_datetime(timeseries_df['Date'])
    
    # # Dealing with dates
    # timeseries_df['Year'] = timeseries_df['Date'].dt.year
    # timeseries_df['Month'] = timeseries_df['Date'].dt.month
    # timeseries_df['Day'] = timeseries_df['Date'].dt.day
    # timeseries_df['DayOfWeek'] = timeseries_df['Date'].dt.dayofweek  # Monday=0, Sunday=6
    # timeseries_df['DayOfYear'] = timeseries_df['Date'].dt.dayofyear
    # timeseries_df['Week'] = timeseries_df['Date'].dt.isocalendar().week
    # timeseries_df['Quarter'] = timeseries_df['Date'].dt.quarter
    
    # # Add readable day names
    # timeseries_df['DayName'] = timeseries_df['Date'].dt.day_name()
    # timeseries_df['MonthName'] = timeseries_df['Date'].dt.month_name()
    
    #All matches with results only.  Used for plots to that show win rates.
    win_count_df = data_for_avg.groupby("Team")["Result"].value_counts().reset_index()
 
    columns_to_keep = ['Date','Team', 'Opp', 'Result']
    match_result_lookup = all_data_df[columns_to_keep]
    
    enhanced_results_summary_df = preproc.enhanced_results_summary(match_result_lookup)
    
    #merged_df = pd.merge(data_for_avg, enhanced_results_summary_df, how = "inner", on = "Team")
    
    merged_df = create_merge_data(data_for_avg)
    
    ########################
    # Create data to track the accuracy 
    # of the random forest prediction
    # model performanceand accuracy
    ########################    
    
    #accuracy_tracking = pd.DataFrame({"Game Week" : ["GW 1", "GW 2", "GW 3", "GW 4", "GW 5", "GW 6", "GW 7"],
     #                                 "Accuracy" : [60, 70, 40, 70, 50, 40, 60],
     #                                 "Running Median" : [60, 65, 60, 65, 60, 55, 60]})
     
    accuracy_tracking, mean_mean = create_accuracy_data() 
    
    ########################
    # Create stadium data
    ######################## 
    
    # stadium_data = pd.read_csv("Data/stadiums.csv")
    
    # all_df = pd.DataFrame(all_data_df["Team"].unique(), columns = ["Team"])
    # idx = all_df[all_df['Team'] == "Nott'ham Forest"].index
    # all_df.loc[idx, 'Team'] = "Nottingham Forest"
    # idx = all_df[all_df['Team'] == "Newcastle Utd"].index
    # all_df.loc[idx, 'Team'] = "Newcastle United"
    
    # all_df_for_Stadium_merge = all_df.copy()
    
    # stadiums_pl = pd.merge(all_df_for_Stadium_merge, stadium_data, how="left", on="Team")
    
    stadiums_pl = create_stadium_data(all_data_df)
    
    #####################################
    # Create country data for location and merging
    ####################################   
    
    countries_df = create_country_data()
    country_codes_df = create_country_code_data()
    
    #####################################
    # Create player data used in tab6 - Players
    ####################################

    # Load and clean up
    #all_players_df = pd.read_csv("Data/players_25_26.csv")
    # all_players_df.drop(["Season", "Comp", "-9999"], axis=1, inplace=True)
    
    # #country_codes_df = pd.read_csv("Data/world.csv")
    
    # all_players_df['Nation'] = all_players_df['Nation'].str.replace(r'[A-Z]{3}', '', regex=True).str.strip()
    
    # all_players_df['Nation'] = all_players_df['Nation'].replace({'eng': 'gb',
    #                                                              'wls': 'gb',
    #                                                              'nir': 'gb',
    #                                                              'sct': 'gb'})
    
    # # Add flag_path URLs
    # all_players_df['flag_path'] = all_players_df['Nation'].apply(
    #     lambda x: f"https://flagcdn.com/64x48/{x}.png"
    # )
    
    # all_players_df['Nation'] = all_players_df['Nation'].str.upper()
    
    # # Add extra stadium data
    # #stadiums_pl = sm.stadium_merge(stadiums_pl)
    
    # # Grab badge from stadiums-pl and add to players
    # all_players_df['Team'] = all_players_df['Team'].replace({"Nott'ham Forest": 'Nottingham Forest'})
    # all_players_df = all_players_df.merge(stadiums_pl[["Team", "Badge"]], how="left", on="Team")
    
    # # Add in a % of total minutes for the season 28*90
    # all_players_df["Perc_Min_Season"] = all_players_df["Min"] / (28 * 90)
    # all_players_df.fillna(value = 0, inplace=True)    
    
    # # Import country location data to be able to build hub and spoke maps for teams and players.
    # # countries_df = pd.read_csv("Data/countries.csv")
    
    # all_players_df = all_players_df.merge(countries_df[['country', 'latitude', 'longitude', 'name']], how="left", left_on='Nation', right_on='country')
    
    all_players_df = create_player_data(stadiums_pl, countries_df)
    
    #######################
    # Match attendance data for Leader boards
    #######################
    # attendance_df = pd.read_csv("Data/attendance_tracking_20250931.csv")
    
    # # Merge in stadium data with attendance
    # attendance_df = attendance_df.merge(stadiums_pl, how="left", on="Team")
    
    attendance_df = create_attendance_date(stadiums_pl)
    
    ######################
    # Load coach data and clean
    ######################
    
    # # Load coach data
    # coaches_df = pd.read_csv("Data/coaches.csv")
    # #coaches_df.isna().sum()
    # coaches_df.drop([" Ref"], axis= 1, inplace=True)
    
    # # Deal with dates
    # # Convert Start and End to datetime
    # coaches_df['Start'] = pd.to_datetime(coaches_df['Start'], format='%d %B %Y', errors='coerce')
    # coaches_df['End'] = pd.to_datetime(coaches_df['End'], format='%d %B %Y', errors='coerce')
    
    # # Deal with NaT in date columns
    # # First, ensure Duration is numeric (in days)
    # coaches_df['Duration'] = pd.to_numeric(coaches_df['Duration'], errors='coerce')
    
    # # Fill missing End dates using Start + Duration
    # mask_missing_end = coaches_df['End'].isna() & coaches_df['Start'].notna()
    # coaches_df.loc[mask_missing_end, 'End'] = (
    #     coaches_df.loc[mask_missing_end, 'Start'] + pd.to_timedelta(coaches_df.loc[mask_missing_end, 'Duration'], unit='D')
    # )
    
    # # Fill missing Start dates using End - Duration
    # mask_missing_start = coaches_df['Start'].isna() & coaches_df['End'].notna()
    # coaches_df.loc[mask_missing_start, 'Start'] = (
    #     coaches_df.loc[mask_missing_start, 'End'] - pd.to_timedelta(coaches_df.loc[mask_missing_start, 'Duration'], unit='D')
    # )
    
    # # Add national flag data.
 
    # # first convert Nationality to 2 char value
    # coaches_df = coaches_df.merge(countries_df[['country', 'latitude', 'longitude', 'name']], how="left", left_on='Nationality', right_on='name')

    # # convert 2 char code to lower case to be able to create the correct link for the flag image.
    # coaches_df['country'] = coaches_df['country'].str.lower()
    
    # # Add flag_path URLs
    # coaches_df['flag_path'] = coaches_df['country'].apply(
    #     lambda x: f"https://flagcdn.com/64x48/{x}.png"
    # )
    
    # # Correct flags for UK countries
    # idx = coaches_df[coaches_df['name'] == "Scotland"].index
    # coaches_df.loc[idx, 'flag_path'] = "https://upload.wikimedia.org/wikipedia/commons/1/10/Flag_of_Scotland.svg"
    
    # idx = coaches_df[coaches_df['name'] == "Wales"].index
    # coaches_df.loc[idx, 'flag_path'] = "https://upload.wikimedia.org/wikipedia/commons/d/dc/Flag_of_Wales.svg"
    
    # idx = coaches_df[coaches_df['name'] == "Northern Ireland"].index
    # coaches_df.loc[idx, 'flag_path'] = "https://upload.wikimedia.org/wikipedia/commons/4/45/Flag_of_Ireland.svg"
    
    # idx = coaches_df[coaches_df['name'] == "Republic of Ireland"].index
    # coaches_df.loc[idx, 'flag_path'] = "https://upload.wikimedia.org/wikipedia/commons/4/45/Flag_of_Ireland.svg"

    
    # # grab team badge image where available.
    # coaches_df = coaches_df.merge(stadiums_pl[["Team", "Badge"]], how="left", on="Team")
    
    # # Remove symbols from names
    # coaches_df['Name'] = coaches_df['Name'].str.replace('‡', '').str.replace('†', '').str.replace('§', '').str.strip()
   
    coaches_df = create_coach_data(countries_df, stadiums_pl) 
    
    ################################################
    # Load Table data
    ################################################    
    table_all_df = create_table_all()
    
    ################################################
    # Create structure of the UX
    ################################################
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Make Prediction", "Data and Model", "About Project", "Analytics", "Maps", "Players", "Leader Boards", "Managers"])
        
    ################################################
    # Contents of tab1 - Make Prediction
    ################################################
    
    
    with tab1:
        st.session_state.active_tab = "Tab 1"
        st.sidebar.empty()
        # Image at top of tab
        st.image("Images/crystal_palace_stadium.jpg")
    
        # Add title and references
        st.title("Premier League Match Predictions")
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
                
                # First collect all data for each team in to sepa"rate dataframes.
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
                
                if (len(tmp_team1) < 0) | (len(tmp_team2) < 0):
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
                    st.write(f"Using {clf_reduced_name}")
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
                    df_transposed.columns = df_transposed.iloc[5]
                   
                    # Drop the 'Team' row since it's now the column names
                    df_transposed = df_transposed.drop(df_transposed.index[5])
    
    
                    df_transposed["Diff"] = df_transposed[team1_name] - df_transposed[team2_name]
                    
                    styled_df = df_transposed.style.apply(ts.highlight_max, axis=1)
                    st.dataframe(styled_df)
                    #st.dataframe(df_transposed)
                    
                    st.text("Team stats __after__ any statistical logic or favoring in order of importance to prediction")
                    
                    df_transposed = combined_avg.transpose()
    
                    # Set the first row (index 0, which is 'Team') as column names
                    df_transposed.columns = df_transposed.iloc[5]
                   
                    # Drop the 'Team' row since it's now the column names
                    df_transposed = df_transposed.drop(df_transposed.index[5])
    
                    df_transposed["Diff"] = df_transposed[team1_name] - df_transposed[team2_name]
                    styled_df = df_transposed.style.apply(ts.highlight_max, axis=1)
                    st.dataframe(styled_df)
                    
                   # Get the statistics (row index)
                    stats = df_transposed.index.tolist()  # ['SoT', 'GF', 'Poss', 'Long_Cmp', 'Succ', 'Blocks']
                    
                    # Get team names (excluding 'Diff' column)
                    teams = [col for col in df_transposed.columns if col != 'Diff']  # ['Team 1', 'Team 2']
                    
                    x = np.arange(len(stats))  # the label locations (one for each stat)
                    width = 0.35  # the width of the bars
                    
                    fig, ax = plt.subplots(layout='constrained', figsize=(10, 5))
                    
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
             "game_week7",
             "game_week8",
             "game_week9",
             "game_week10",
             "game_week11",
             "game_week12",
             "game_week13",
             "game_week14",
             "game_week15",
             "game_week16",
             "game_week17",
             "game_week18",
             "game_week19",
             "game_week20",
             "game_week21",
             "game_week22",
             ),  key="gw_num_pick")
        
        
        # Create dataframes for each game week containing Match ID, Actual  Result, Predicted Result
       
        # Game week 1
        # gw_1_actuals_list = [["-", 1,"Liverpool", "Bournemouth", "4-2", "Home Win", "Home Win"],
        #                   ["-", 2,"Aston Villa", "Newcastle", "0-0", "Tie", "Home Win"],
        #                   ["-", 3,"Brighton", "Fulham", "1-1", "Tie", "Tie"],
        #                   ["-", 4,"Sunderland", "West Ham", "3-0", "Home Win", "Home Win"],
        #                   ["-", 5,"Spurs", "Burnley", "3-0", "Home Win", "Home Win"],
        #                   ["-", 6,"Wolves", "Man City", "0-4", "Away Win", "Away Win"],
        #                   ["-", 7,"Nott'm Forest", "Brentford", "3-1", "Home Win", "Tie"], 
        #                   ["-", 8,"Chelsea", "Crystal Palace", "0-0", "Tie", "Home Win"],
        #                   ["-", 9,"Man Utd", "Arsenal", "0-1", "Away Win", "Away Win"],
        #                   ["-", 10,"Leeds United", "Everton", "1-0", "Home Win", "Tie"]]
        # gw_1_actuals = pd.DataFrame(gw_1_actuals_list, columns=["Date", "match ID","Home","Away", "Score", "Result", "Predicted"])
    
        # # Game week 2    
        # gw_2_actuals_list = [["-", 1,"West Ham", "Chelsea", "1-5", "Away Win", "Away Win"],
        #                   ["-", 2,"Man City", "Spurs", "0-2", "Away Win", "Home Win"],
        #                   ["-", 3,"Bournemouth", "Wolves", "1-0", "Home Win", "Home Win"],
        #                   ["-", 4,"Brentford", "Aston Villa", "1-0", "Home Win", "Away Win"],
        #                   ["-", 5,"Burnley", "Sunderland", "2-0", "Home Win", "Home Win"],
        #                   ["-", 6,"Arsenal", "Leeds United", "5-0", "Home Win", "Home Win"],
        #                   ["-", 7,"Crystal Palace", "Nott'm Forest", "1-1", "Tie", "Tie"], 
        #                   ["-", 8,"Everton", "Brighton", "2-0", "Home Win", "Home Win"],
        #                   ["-", 9,"Fulham", "Man Utd", "1-1", "Tie", "Away Win"],
        #                   ["-", 10,"Newcastle", "Liverpool", "2-3", "Away Win", "Away Win"]]
        # gw_2_actuals = pd.DataFrame(gw_2_actuals_list, columns=["Date", "match ID","Home","Away", "Score", "Result", "Predicted"])
    
        # # game week 3
        # gw_3_actuals_list = [["-", 1,"Chelsea", "Fulham", "2-0", "Home Win", "Home Win"],
        #                   ["-", 2, "Man Utd", "Burnley", "3-2", "Home Win", "Home Win"],
        #                   ["-", 3, "Sunderland", "Brentford", "2-1", "Home Win", "Home Win"],
        #                   ["-", 4, "Spurs", "Bournemouth", "0-1", "Away Win", "Home Win"],
        #                   ["-", 5, "Wolves", "Everton", "2-3", "Away Win", "Home Win"],
        #                   ["-", 6, "Leeds United", "Newcastle", "0-0", "Tie", "Away Win"],
        #                   ["-", 7, "Brighton", "Man City", "2-1", "Home Win", "Away Win"], 
        #                   ["-", 8, "Nott'm Forest", "West Ham", "0-3", "Away Win", "Tie"],
        #                   ["-", 9, "Liverpool", "Arsenal", "1-0", "Home Win", "Home Win"],
        #                   ["-", 10,"Aston Villa", "Crystal Palace", "0-3", "Away Win", "Home Win"]]
        # gw_3_actuals = pd.DataFrame(gw_3_actuals_list, columns=["Date", "match ID","Home","Away", "Score", "Result", "Predicted"])
    
        # # game week 4
        # gw_4_actuals_list = [["-", 1,"Arsenal", "Nott'm Forest", "3-0", "Home Win", "Home Win"],
        #                 ["-", 2,"Bournemouth", "Brighton", "2-1", "Home Win", "Away Win"],
        #                 ["-", 3,"Crystal Palace", "Sunderland", "0-0", "Tie", "Tie"],
        #                 ["-", 4,"Everton", "Aston Villa", "0-0", "Tie", "Away Win"],
        #                 ["-", 5,"Fulham", "Leeds United", "1-0", "Home Win", "Home Win"],
        #                 ["-", 6,"Newcastle", "Wolves", "1-0", "Home Win", "Home Win"],
        #                 ["-", 7,"West Ham", "Spurs", "0-3", "Away Win", "Away Win"],
        #                 ["-", 8,"Brentford", "Chelsea", "2-2", "Tie", "Away Win"],
        #                 ["-", 9,"Burnley", "Liverpool", "0-1", "Away Win", "Away Win"],
        #                 ["-", 10,"Man City", "Man Utd", "3-0", "Home Win", "Home Win"]]
        # gw_4_actuals = pd.DataFrame(gw_4_actuals_list, columns=["Date", "match ID","Home","Away", "Score", "Result", "Predicted"])
        
        # # game week 5
        # gw_5_actuals_list = [["Sat 20 Sep 04:30","Liverpool", "Everton", "2-1", "Home Win", "Home Win"],
        #                   ["Sat 20 Sep 07:00", "Brighton", "Spurs", "2-2", "Tie", "Away Win"],
        #                   ["Sat 20 Sep 07:00", "Burnley", "Nott'm Forest", "1-1", "Tie", "Tie"],
        #                   ["Sat 20 Sep 07:00", "West Ham", "Crystal Palace", "1-2",  "Away Win", "Away Win"],
        #                   ["Sat 20 Sep 07:00", "Wolves", "Leeds Utd", "1-3",  "Away Win", "Away Win"],
        #                   ["Sat 20 Sep 09:30", "Man Utd", "Chelsea", "2-1",  "Home Win", "Away Win"],
        #                   ["Sat 20 Sep 12:00", "Fulham", "Brentford", "3-1",  "Home Win", "Home Win"], 
        #                   ["Sun 21 Sep 06:00", "Bournemouth", "Newcastle", "0-0",  "Tie", "Away Win"],
        #                   ["Sun 21 Sep 06:00", "Sunderland", "Aston Villa", "1-1",  "Tie", "Away Win"],
        #                   ["Sun 21 Sep 08:30", "Arsenal", "Man City", "1-1",  "Tie", "Away Win"]]
        # gw_5_actuals = pd.DataFrame(gw_5_actuals_list, columns=["Date", "Home","Away", "Score", "Result", "Predicted"])
      
        # # game week 5
        # gw_5_actuals_list = [["Sat 20 Sep 04:30","Liverpool", "Everton", "2-1", "Home Win", "Home Win"],
        #                   ["Sat 20 Sep 07:00", "Brighton", "Spurs", "2-2", "Tie", "Away Win"],
        #                   ["Sat 20 Sep 07:00", "Burnley", "Nott'm Forest", "1-1", "Tie", "Tie"],
        #                   ["Sat 20 Sep 07:00", "West Ham", "Crystal Palace", "1-2",  "Away Win", "Away Win"],
        #                   ["Sat 20 Sep 07:00", "Wolves", "Leeds Utd", "1-3",  "Away Win", "Away Win"],
        #                   ["Sat 20 Sep 09:30", "Man Utd", "Chelsea", "2-1",  "Home Win", "Away Win"],
        #                   ["Sat 20 Sep 12:00", "Fulham", "Brentford", "3-1",  "Home Win", "Home Win"], 
        #                   ["Sun 21 Sep 06:00", "Bournemouth", "Newcastle", "0-0",  "Tie", "Away Win"],
        #                   ["Sun 21 Sep 06:00", "Sunderland", "Aston Villa", "1-1",  "Tie", "Away Win"],
        #                   ["Sun 21 Sep 08:30", "Arsenal", "Man City", "1-1",  "Tie", "Away Win"]]
        # gw_5_actuals = pd.DataFrame(gw_5_actuals_list, columns=["Date", "Home","Away", "Score", "Result", "Predicted"])
    
        # # game week 6
        # gw_6_actuals_list = [["Sat 27 Sep 04:30","Brentford", "Man Utd", "3-1", "Home Win", "Away Win"],
        #                   ["Sat 27 Sep 07:00", "Chelsea", "Brighton",  "1-3", "Away Win", "Home Win"],
        #                   ["Sat 27 Sep 07:00", "Crystal Palace", "Liverpool", "2-1", "Home Win", "Away Win"],
        #                   ["Sat 27 Sep 07:00", "Leeds Utd", "Bournemouth",  "2-2", "Tie", "Tie"],
        #                   ["Sat 27 Sep 07:00", "Man City", "Burnley",  "5-1", "Home Win", "Home Win"],
        #                   ["Sat 27 Sep 09:30", "Nott'm Forest", "Sunderland",  "0-1", "Away Win", "Tie"],
        #                   ["Sat 27 Sep 12:00", "Spurs", "Wolves",  "1-1", "Tie", "Home Win"], 
        #                   ["Sun 28 Sep 06:00", "Aston Villa", "Fulham",  "3-1", "Home Win", "Home Win"],
        #                   ["Sun 28 Sep 08:30", "Newcastle", "Arsenal",  "1-2", "Away Win", "Away Win"],
        #                   ["Mon 29 Sep 12:00", "Everton", "West Ham",  "1-1", "Tie", "Away Win"]]
        # gw_6_actuals = pd.DataFrame(gw_6_actuals_list, columns=["Date", "Home","Away", "Score", "Result", "Predicted"])
        
        # # game week 7
        # gw_7_actuals_list = [["Fri 03 Oct 04:30","Bournwmouth", "Fulham", "3-1", "Home Win", "Tie"],
        #                   ["Sat 04 Oct 12:00", "Leeds", "Spurs",  "1-2", "Away Win", "Away Win"],
        #                   ["Sat 04 Oct 07:00", "Arsenal", "West Ham", "2-0", "Home Win", "Home Win"],
        #                   ["Sat 04 Oct 07:00", "Man Utd", "Sunderland",  "2-0", "Home Win", "Home Win"],
        #                   ["Sat 04 Oct 09:30", "Chelsea", "Liverpool",  "2-1", "Home Win", "Away Win"],
        #                   ["Sun 05 Oct 06:00", "Aston Villa", "Burnley",  "2-1", "Home Win", "Home Win"],
        #                   ["Sun 05 Oct 06:00", "Everton", "Crystal Palace",  "2-1", "Home Win", "Tie"], 
        #                   ["Sun 05 Oct 06:00", "Newcastle", "Nott'm Forest",  "2-0", "Home Win", "Home Win"],
        #                   ["Sun 05 Oct 06:00", "Wolves", "Brighton",  "1-1", "Tie", "Away Win"],
        #                   ["Sun 05 Oct 08:30", "Brentford", "Man City",  "0-1", "Away Win", "Away Win"]]
        # gw_7_actuals = pd.DataFrame(gw_7_actuals_list, columns=["Date", "Home","Away", "Score", "Result", "Predicted"])
        
        # gw_7_actuals.info()
        
        # # game week 7
        # gw_8_actuals_list = [["Sat 18 Oct 04:30","Nott'ham Forest", "Chelsea", np.nan, np.nan, "Away Win"],
        #                   ["Sat 18 Oct 07:00", "Brighton", "Newcastle",  np.nan, np.nan, "Away Win"],
        #                   ["Sat 18 Oct 07:00", "Burnley", "Leeds", np.nan, np.nan, "Tie"],
        #                   ["Sat 18 Oct 07:00", "Crystal Palace", "Bournemouth",  np.nan, np.nan, "Tie"],
        #                   ["Sat 18 Oct 07:00", "Man City", "Everton",  np.nan, np.nan, "Home Win"],
        #                   ["Sat 18 Oct 07:00", "Sunderland", "Wolves",  np.nan, np.nan, "Tie"],
        #                   ["Sat 18 Oct 09:30", "Fulham", "Arsenal",  np.nan, np.nan, "Away Win"], 
        #                   ["Sun 19 Oct 06:00", "Spurs", "Aston Villa",  np.nan, np.nan, "Home Win"],
        #                   ["Sun 19 Oct 08:30", "Liverpool", "Man Utd",  np.nan, np.nan, "Home Win"],
        #                   ["Mon 20 Oct 12:00", "West Ham", "Brentoford",  np.nan, np.nan, "Tie"]]
        # gw_8_actuals = pd.DataFrame(gw_8_actuals_list, columns=["Date", "Home","Away", "Score", "Result", "Predicted"])
            
        # Only use to post special notes about matches.  Otherwise keep False.
        include_note = True
        gw_note = "game_week8"
        note = "The Crystal Palace V Bournemouth match will be close with Bournemouth possibly pulling out a Win. \nPrediction is a Tie, but I am watching this one closely for Bournemouth Win. \nSame goes for the Sunderland v Wolves match.  Sunderland is playing wel."
        # mapping of game selection text to the correct dataframe
        # actuals_week_mapping = {
        #     "game_week1": gw_1_actuals,
        #     "game_week2": gw_2_actuals,
        #     "game_week3": gw_3_actuals,
        #     "game_week4": gw_4_actuals, 
        #     "game_week5": gw_5_actuals,
        #     "game_week6": gw_6_actuals,
        #     "game_week7": gw_7_actuals,
        #     "game_week8": gw_8_actuals
        # }
        
        # Display Actual information
        
        st.subheader("Schedule with Actual VS. Predicted")
        
        # Get the selected DataFrame and display it
        selected_dataframe = getattr(create_actuals, gw_num_pick)()
        
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
                
                accuracy_tracking = accuracy_tracking.sort_values(
                    'Game Week',
                    key=lambda x: x.str.extract('(\d+)', expand=False).astype(int)
                    )
                accuracy_tracking_chart = accuracy_tracking.set_index("Game Week")
                st.line_chart(accuracy_tracking_chart)
                st.write(f"Mean of Mean of the Accuracy: {mean_mean}")
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
        # table_all_df = pd.read_csv("Data/tables_all.csv")
        
        # create a dataframe for each game week from table_all_df and selecting  on gw_num.
        # I am not sure why this needs to be here instead of making use of the create_table_data().
        table_1_game_df = table_all_df[table_all_df["Pl"] == 1]
        table_2_game_df = table_all_df[table_all_df["Pl"] == 2]
        table_3_game_df = table_all_df[table_all_df["Pl"] == 3] 
        table_4_game_df = table_all_df[table_all_df["Pl"] == 4] 
        table_5_game_df = table_all_df[table_all_df["Pl"] == 5]
        table_6_game_df = table_all_df[table_all_df["Pl"] == 6]    
        table_7_game_df = table_all_df[table_all_df["Pl"] == 7]   
        table_8_game_df = table_all_df[table_all_df["Pl"] == 8] 
        table_9_game_df = table_all_df[table_all_df["Pl"] == 9] 
        table_10_game_df = table_all_df[table_all_df["Pl"] == 10] 
        table_11_game_df = table_all_df[table_all_df["Pl"] == 11]
        table_12_game_df = table_all_df[table_all_df["Pl"] == 12]
        table_13_game_df = table_all_df[table_all_df["Pl"] == 13]
        table_14_game_df = table_all_df[table_all_df["Pl"] == 14]
        table_15_game_df = table_all_df[table_all_df["Pl"] == 15]
        table_16_game_df = table_all_df[table_all_df["Pl"] == 16]
        table_17_game_df = table_all_df[table_all_df["Pl"] == 17]
        table_18_game_df = table_all_df[table_all_df["Pl"] == 18]
        table_19_game_df = table_all_df[table_all_df["Pl"] == 19]
        table_20_game_df = table_all_df[table_all_df["Pl"] == 20]
        table_21_game_df = table_all_df[table_all_df["Pl"] == 21]
        
        # Mapping for selecte gameweek to correct table dataframe
        table_mapping = {
            "post game week 1": table_1_game_df,
            "post game week 2": table_2_game_df,
            "post game week 3": table_3_game_df,
            "post game week 4": table_4_game_df,
            "post game week 5": table_5_game_df,
            "post game week 6": table_6_game_df,
            "post game week 7": table_7_game_df,
            "post game week 8": table_8_game_df,
            "post game week 9": table_9_game_df,
            "post game week 10": table_10_game_df,
            "post game week 11": table_11_game_df,
            "post game week 12": table_12_game_df,
            "post game week 13": table_13_game_df,
            "post game week 14": table_14_game_df,
            "post game week 15": table_15_game_df,
            "post game week 16": table_16_game_df,
            "post game week 17": table_17_game_df,
            "post game week 18": table_18_game_df,
            "post game week 19": table_19_game_df,
            "post game week 20": table_20_game_df,
            "post game week 21": table_21_game_df
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
             "post game week 20",
             "post game week 21"
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
        
        # show form guide
        import Data.create_actuals as actuals
        import Plot.form_table as form_table
        
        # Create actuals and Table data
        all_actuals_list = actuals.create_all_actuals()
        table_all_df = create_table_all()
        
        # Create Plotly visual
        fig = form_table.create_enhanced_team_form_table(table_all_df, all_actuals_list)
        
        # Display visual
        st.plotly_chart(fig, width="stretch")                   
    
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
        # st.subheader("Performance Dashboard")
        # dashboard_fig = pp.create_football_performance_dashboard(timeseries_df)
        # st.pyplot(dashboard_fig)
        
        # Team performance
        st.subheader("Team Performance Chart")
        team_performance_fig =  pp.team_performance(merged_df)
        tab1, tab2 = st.tabs(["Chart", "Dataframe"])
        tab1.pyplot(team_performance_fig)
        tab2.dataframe(merged_df, height=250, width=200)
    
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
        st.write("A form table shows the results of the last 6 matches in order of the latest Table positionining")
        
        # import Data.create_actuals as actuals
        # import Plot.form_table as form_table
        
        # # Create actuals and Table data
        # all_actuals_list = actuals.create_all_actuals()
        # table_all_df = create_table_all()
        
        # # Create Plotly visual
        # fig = form_table.create_enhanced_team_form_table(table_all_df, all_actuals_list)
        
        # # Display visual
        # st.plotly_chart(fig, use_container_width=True)
        
    
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
                    
                    # display diveder
                    st.html('<hr style="border: none; height: 3px; background-color: #808080;">')
                    
                    # Create a bar chart visualization
                    st.subheader("Number of Players by Country")
                    # Sort by player count (descending)
                    country_counts_sorted = country_counts.sort_values('Player_Count', ascending=False)
                    
                    fig = px.bar(
                        country_counts_sorted,
                        x='Country',
                        y='Player_Count',
                        labels={
                            'Country': 'Countries Represented by Players',
                            'Player_Count': 'Number of Players'
                        }
                    )
                    
                    fig.update_layout(
                        xaxis_title="Countries Represented by Players",
                        yaxis_title="Number of Players",
                        showlegend=False
                    )
                    
                    fig.update_xaxes(tickangle=-45)
                    
                    st.plotly_chart(fig, width="stretch")
                    
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
        
        team_options = ['All'] + sorted(all_players_df['Team'].unique().tolist())
        position_options = ['All'] + sorted(all_players_df['Pos'].unique().tolist())
        country_options = ['All'] + sorted(all_players_df['name'].dropna().unique().tolist())

        with st.form("filter_form"):
            st.write("**Make selections to filter players and click the Filter Data button to see the table.**")
            st.write("**The Team and Position** filter have ***All*** options to see all in that filter.")
            st.write("**To not filter by a stat** simply set the minimum value to 0 for any stat.")
            st.write("**The columns on the table** also allow sorting ascending or descending by clicking on the the 3 dots next to the name.")
            team_filter = st.selectbox("Select Team", team_options)
            position_filter = st.selectbox("Select Position", position_options)
            country_filter = st.selectbox("Select Country From", country_options)
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
        
        player_tab, team_tab, fans_tab = st.tabs(["Player", "Team", "Fans"])
        
        
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
            
        with fans_tab:
            
    
            # Calculate utilization rate
            attendance_df['Utilization'] = (attendance_df['Attendance'] / attendance_df['Capacity']) * 100
            
            # Get average utilization by team
            avg_utilization = attendance_df.groupby('Team')['Utilization'].mean().sort_values(ascending=False).reset_index()
            
            # merge avg util to stadiums
            stadiums_pl = stadiums_pl.merge(avg_utilization[["Team", "Utilization"]], how="left", on="Team")
            
            #get table data
                
            # create a dataframe for each game week from table_all_df and selecting  on gw_num 
            table_all_df = pd.read_csv("Data/tables_all.csv")
    
            table_6_game_df = table_all_df[table_all_df["Pl"] == 6] 
            
            idx = table_6_game_df[table_6_game_df['Team'] == "Tottenham Hotspur"].index
            table_6_game_df.loc[idx, 'Team'] = "Tottenham"
            
            idx = table_6_game_df[table_6_game_df['Team'] == "AFC Bournemouth"].index
            table_6_game_df.loc[idx, 'Team'] = "Bournemouth"
            
            idx = table_6_game_df[table_6_game_df['Team'] == "Brighton & Hove Albion"].index
            table_6_game_df.loc[idx, 'Team'] = "Brighton"
            
            idx = table_6_game_df[table_6_game_df['Team'] == "Manchester United"].index
            table_6_game_df.loc[idx, 'Team'] = "Manchester Utd"
            
            idx = table_6_game_df[table_6_game_df['Team'] == "West Ham United"].index
            table_6_game_df.loc[idx, 'Team'] = "West Ham"
            
            idx = table_6_game_df[table_6_game_df['Team'] == "Wolverhampton Wanderers"].index
            table_6_game_df.loc[idx, 'Team'] = "Wolves"
            
            #merge to stadiums
            stadiums_pl = stadiums_pl.merge(table_6_game_df[["Team", "Pos"]], how="left", on="Team")
            
            # remove teams that are not on the table
            stadiums_pl = stadiums_pl.dropna(subset=['Pos'])
            
            # Merge in win rate data.
            idx = team_stats[team_stats['Team'] == "Nott'ham Forest"].index
            team_stats.loc[idx, 'Team'] = "Nottingham Forest"
            
            idx = team_stats[team_stats['Team'] == "Newcastle Utd"].index
            team_stats.loc[idx, 'Team'] = "Newcastle United"
            
            stadiums_pl = stadiums_pl.merge(team_stats[["Team", "win_rate"]], how="left", on="Team")
            
            
            # Display stadium_pl
            st.write("All teams in 2025/26 season, capacity, average utilization of capacity")
            st.dataframe(stadiums_pl, column_order=["Pos", "Team", "win_rate", "Name", "Attendance", "Capacity", "Utilization"], column_config={        
                "Pos": st.column_config.NumberColumn(
                    "Table Pos.",
                    help="Latest table ranking"
                ),
                "Team": st.column_config.TextColumn(
                    "Team", 
                    help="Current team the player belongs to"
                ),
                "win_rate": st.column_config.ProgressColumn(
                    "Win Rate",
                    help="All time win rate"
                ),
                "Name": st.column_config.TextColumn(
                    "Stadium", 
                    help="Name of the stadium"
                ),
                "Attendance": st.column_config.NumberColumn(
                    "Avg. Addenance", 
                    help="The average of all matches played at this stadium so far in 20225"
                ),
                "Capacity": st.column_config.NumberColumn(
                    "Capacity", 
                    help="The current official capacity of the stadium"
                ),
                "Utilization": st.column_config.NumberColumn(
                    "Percent Utilization", 
                    help="Average percent utilization of allowed attendance",
                    format="%.2f"  # 2 decimal places
                ),
            },hide_index=True)
            
            fig = px.bar(avg_utilization, 
                         x="Team", 
                         y="Utilization",
                         title='Average Stadium Utilization Rate by Team',
                         labels={'x': 'Team', 'y': 'Utilization (%)'},
                         color_discrete_sequence=['steelblue'])
            
            fig.update_layout(
                title_font_size=14,
                xaxis_tickangle=-45,
                yaxis_gridcolor='rgba(0,0,0,0.3)',
                height=600,
                width=1200
                )
            
            st.plotly_chart(fig)
            
            fig = px.scatter(
                stadiums_pl,
                x='win_rate',
                y='Utilization',
                text='Team',  # Shows team names on hover
                title='Win Rate vs Stadium Utilization',
                labels={
                    'stadium_utilization': 'Stadium Utilization (%)',
                    'win_rate': 'Win Rate'
                }
            )
            
            # Optional: add team labels to points
            fig.update_traces(textposition='top center')
            
            st.plotly_chart(fig)
            # Plot 5: Top/Bottom Attendance
            import plotly.graph_objects as go
            import plotly.express as px
            
            avg_attendance = attendance_df.groupby('Team')['Attendance'].mean().sort_values(ascending=True)
            
            fig = go.Figure(data=[
                go.Bar(x=avg_attendance.values, y=avg_attendance.index, orientation='h',
                       marker_color='steelblue')
            ])
            fig.update_layout(
                title='Average Attendance by Team',
                xaxis_title='Average Attendance',
                yaxis_title='Team',
                height=600,
                showlegend=False
            )
    
            st.plotly_chart(fig)
            
            # # Highest attendanded match
            max_attend_match = attendance_df.loc[attendance_df['Attendance'].idxmax()].reset_index()
            
            df_transposed = max_attend_match.transpose()
    
    
            # Set the first row (index 0, which is 'Team') as column names
            df_transposed.columns = df_transposed.iloc[0]
           
            # Drop the 'Team' row since it's now the column names
            df_transposed = df_transposed.drop(df_transposed.index[0])
    
            # st.write("__2025 Match with the highest attendance:__")

            # st.dataframe(df_transposed, column_order=["Date", "Team", "Opponent", "Attendance", "Capacity", "Utilization"],  column_config={        
            #     "Pos": st.column_config.TextColumn(
            #         "Position",
            #         help="Latest table ranking"
            #     ),
            #     "Date": st.column_config.TextColumn(
            #         "Date (DD/MM/YYYY)", 
            #         help="Date of the match"
            #     ),
            #     "Team": st.column_config.TextColumn(
            #         "Home TEam", 
            #         help="Home Team" 
            #     ),
            #     "Opponent": st.column_config.TextColumn(
            #         "Away Team", 
            #         help="Away Team"
            #     ),
            #     "Attendance": st.column_config.TextColumn(
            #         "Avg. Addenance", 
            #         help="The average of all matches played at this stadium so far in 20225"
            #     ),
            #     "Capacity": st.column_config.TextColumn(
            #         "Capacity", 
            #         help="The current official capacity of the stadium"
            #     ),
            #     "Utilization": st.column_config.TextColumn(
            #         "Percent Utilization", 
            #         help="Average percent utilization of allowed attendance"
            #     ),
            # },hide_index=True)
            
            
            import plotly.graph_objects as go
            import plotly.express as px
            
      
            
            # Plot 6: Home vs Away Attendance
            # Home matches: Team appears in 'Team' column
            home_attendance = attendance_df.groupby('Team')['Attendance'].mean()
            
            # Away matches: Team appears in 'Opponent' column
            away_attendance = attendance_df.groupby('Opponent')['Attendance'].mean()
            
            # Combine both (only for teams that appear in both)
            comparison = pd.DataFrame({
                'Home': home_attendance,
                'Away': away_attendance
            }).dropna()
            
            fig = go.Figure(data=[
                go.Bar(name='Home', x=comparison.index, y=comparison['Home'], 
                       marker_color='steelblue'),
                go.Bar(name='Away', x=comparison.index, y=comparison['Away'], 
                       marker_color='coral')
            ])
            fig.update_layout(
                title='Average Attendance: Home vs Away',
                xaxis_title='Team',
                yaxis_title='Average Attendance',
                barmode='group',
                height=500,
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig)
            
            # Plot 7: Big vs Small Clubs (by capacity)
            # Define size categories based on capacity
            attendance_df['Stadium_Size'] = pd.cut(attendance_df['Capacity'], 
                                                    bins=[0, 30000, 50000, 100000],
                                                    labels=['Small (<30k)', 'Medium (30-50k)', 'Large (>50k)'])
            
            size_utilization = attendance_df.groupby('Stadium_Size')['Utilization'].mean()
            
            fig = go.Figure(data=[
                go.Bar(x=size_utilization.index, y=size_utilization.values,
                       marker_color=['#3b82f6', '#8b5cf6', '#ec4899'])
            ])
            fig.update_layout(
                title='Stadium Utilization by Stadium Size',
                xaxis_title='Stadium Size',
                yaxis_title='Average Utilization (%)',
                height=500,
                showlegend=False
            )
            st.plotly_chart(fig)
            
            # Print some insights
            st.subheader("=== Team Comparison Insights ===")
            st.write(f"\n__Highest average attendance:__ {avg_attendance.idxmax()} ({avg_attendance.max():,.0f})")
            st.write(f"__Lowest average attendance:__ {avg_attendance.idxmin()} ({avg_attendance.min():,.0f})")
            
            st.subheader("\n=== Home Advantage ===")
            comparison['Difference'] = comparison['Home'] - comparison['Away']
            st.write(f"\n__Biggest home crowd advantage:__ {comparison['Difference'].idxmax()}")
            st.write(f"  Home: {comparison.loc[comparison['Difference'].idxmax(), 'Home']:,.0f}")
            st.write(f"  Away: {comparison.loc[comparison['Difference'].idxmax(), 'Away']:,.0f}")
            
            st.subheader("\n=== Stadium Size Analysis ===")
            st.write(size_utilization)
          
            
    ################################################
    # Contents of tab7 - Managers
    ################################################
           
    
    with tab8:  
        import streamlit as st
        import plotly.express as px
        
        #############################
        # Display Hub and Spoke map of all 
        #countries coaches are from to the 
        # center of England.
        ##############################

        # Origin is the hub.
        origin = countries_df[countries_df['country'] == "GB"][['latitude', 'longitude', 'name']].iloc[0]
        origin_dict = {'lat': origin['latitude'], 'lon': origin['longitude'], 'name': origin['name']}
        
        # Destinations are the centroid of the countries coaches are from.
        destinations = coaches_df[['country', 'Name', 'latitude', 'longitude', 'name']].drop_duplicates(subset='Name')
        destinations_dict = destinations.rename(columns={'latitude': 'lat', 'longitude': 'lon', 'Name': 'target_name'}).to_dict('records')
        if 'Pep Guardiola' in destinations_dict:
            print("found")
        
        st.title("Home country of each coach who managed a Premier League team")
        
        hub.create_hub_spoke_map(origin_dict, destinations_dict, coaches=True)
        
        st.subheader("Nationality Distribution of Team Managers")
        coach_nat_count = coaches_df.groupby('name')['Name'].nunique().sort_values(ascending=False).reset_index()
        coach_nat_count.columns = ['name', 'count']
        
        fig = px.bar(coach_nat_count, 
                     x='count', 
                     y='name',
                     orientation='h',
                     labels={'count': 'Count of Coaches', 'name': 'Country'})
        fig.update_layout(height=max(600, 800),
                          yaxis={'categoryorder':'total ascending'})  # or 'total descending'
        st.plotly_chart(fig)
        
        ################################
        # Average tenure by nationality
        ################################
        st.subheader("Nationality Distribution of Team Managers and Average Tenure")
        avg_tenure = coaches_df.groupby('Nationality')['Duration'].mean().sort_values(ascending=False).reset_index()
        
        fig = px.bar(avg_tenure,
                     x='Nationality',
                     y='Duration',
                     labels={'Nationality': 'Country', 'Duration': 'Average Days on a Team'})
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig)

        
        ###############################
        # Timeline of coaching tenures
        ###############################
        # Get the list of teams from stadiums_pl
        teams_to_include = stadiums_pl["Team"].tolist()
        
        # Filter coaches_df to only these teams
        coaches_filtered = coaches_df[coaches_df['Team'].isin(teams_to_include)]
        
        # Sort by team and start date
        coaches_sorted = coaches_filtered.sort_values(['Team', 'Start'])
        
        # Remove any rows with missing dates
        coaches_sorted = coaches_sorted.dropna(subset=['Start', 'End'])
        
        # Create the plot with adjusted height
        fig = px.timeline(
            coaches_sorted, 
            x_start='Start', 
            x_end='End', 
            y='Team',
            color='Name',
            title='Team Coaching Timeline (gaps indicate timeframes with multiple coaches with short tenures)'
        )
        
        # Adjust the height based on number of teams (e.g., 30 pixels per team)
        num_teams = coaches_sorted['Team'].nunique()
        fig.update_layout(
            height=max(600, num_teams * 30),  # Minimum 600px, or 30px per team
            yaxis={'categoryorder': 'category ascending'}
        )
        
        st.plotly_chart(fig, width="stretch")
    
        
        # Distribution of tenure lengths
        #fig = px.histogram(coaches_sorted, x='Duration', nbins=20, title='Distribution of Coach Tenure')
        #st.plotly_chart(fig)
        
        ####################
        # Coach journey!!!
        ####################
        
        # Find coaches who have managed multiple teams
        coach_team_counts = coaches_df.groupby('Name')['Team'].nunique()
        multi_team_coaches = coach_team_counts[coach_team_counts > 1].sort_values(ascending=False)
        
        # Create a selectbox for the user to choose a coach
        st.subheader("Coach Career Journey")
        selected_coach = st.selectbox(
            "Select a coach to see their career journey:",
            options=multi_team_coaches.index.tolist(),
            format_func=lambda x: f"{x} ({multi_team_coaches[x]} teams)"
        )
        with st.spinner("Wait for it...", show_time=True):
        
            # Filter data for the selected coach
            coach_journey = coaches_df[coaches_df['Name'] == selected_coach].copy()
            coach_journey = coach_journey.sort_values('Start')
            
            # Display summary stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                country_name = coach_journey['name'].iloc[0]
                flag_path = coach_journey['flag_path'].iloc[0]
                st.image(flag_path, caption=country_name, width=100)
            with col2:
                st.metric("Total Teams", coach_journey['Team'].nunique())
            with col3:
                st.metric("Total Stints", len(coach_journey))
            with col4:
                total_days = coach_journey['Duration'].sum()
                st.metric("Total Years", f"{total_days/365.25:.1f} years")
                
            # Create timeline visualization
            fig = px.timeline(
                coach_journey,
                x_start='Start',
                x_end='End',
                y='Team',
                color='Team',
                title=f"{selected_coach}'s Coaching Career Timeline",
                hover_data=['Nationality', 'Duration']
            )
            
            fig.update_layout(
                height=max(400, len(coach_journey) * 50),
                showlegend=False,
                yaxis={'categoryorder': 'array', 'categoryarray': coach_journey.sort_values('Start')['Team'].tolist()}
            )
            
            st.plotly_chart(fig, width="stretch")
            
            # Show detailed table
            st.subheader("Career Details")
            display_cols = ['Team', 'Start', 'End', 'Duration', 'Nationality']
            st.dataframe(coach_journey, column_order=["Badge", "Team", "Start", "End", "Duration"], column_config={
                "flag_path": st.column_config.ImageColumn(
                            "Nationality",
                            help="Flag of the choach's nationality"
                ),
                "Badge": st.column_config.ImageColumn(
                            "Team",
                            help="Badge of the teaam"
                ),
            }, width=200, hide_index=True)
        
                