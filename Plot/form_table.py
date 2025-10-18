import plotly.graph_objects as go
import pandas as pd
import numpy as np

def get_team_last_6_results(team_name, all_actuals_dfs):
    """
    Get the last 6 match results for a team from multiple game week dataframes.
    
    Parameters:
    - team_name: Name of the team
    - all_actuals_dfs: List of dataframes (e.g., [gw_1_actuals, gw_2_actuals, ...])
    
    Returns:
    - List of results ['W', 'D', 'L'] for last 6 matches
    """
    
    """Get last 6 results with team name normalization."""
    normalized_name = normalize_team_name(team_name)
    results = []
    
    for gw_df in reversed(all_actuals_dfs):
        # Normalize team names in the dataframe
        gw_df_normalized = gw_df.copy()
        gw_df_normalized['Home'] = gw_df_normalized['Home'].apply(normalize_team_name)
        gw_df_normalized['Away'] = gw_df_normalized['Away'].apply(normalize_team_name)
        
        # Check home matches
        home_matches = gw_df_normalized[gw_df_normalized['Home'] == normalized_name]
        for _, match in home_matches.iterrows():
            results.append(match['Result'])
            if len(results) >= 6:
                break
        
        if len(results) >= 6:
            break
            
        # Check away matches
        away_matches = gw_df_normalized[gw_df_normalized['Away'] == normalized_name]
        for _, match in away_matches.iterrows():
            if match['Result'] == 'Home Win':
                results.append('L')
            elif match['Result'] == 'Away Win':
                results.append('W')
            else:
                results.append('D')
            
            if len(results) >= 6:
                break
        
        if len(results) >= 6:
            break
    
    results = results[:6][::-1]
    while len(results) < 6:
        results.insert(0, '')
    
    return results

def create_team_form_table(table_df, all_actuals_dfs):
    """
    Create a Plotly figure showing team standings with last 6 match results.
    
    Parameters:
    - table_df: DataFrame with columns [Pos, Team, Pl, W, D, L, GF, GA, GD, Pts]
    - all_actuals_dfs: List of game week dataframes [gw_1_actuals, gw_2_actuals, ...]
    """
    
    # Get the latest game week data
    latest_gw = table_df[table_df['Pl'] == table_df['Pl'].max()].copy()
    latest_gw = latest_gw.sort_values('Pos').reset_index(drop=True)
    
    # Get last 6 results for each team
    form_data = []
    for _, row in latest_gw.iterrows():
        team_results = get_team_last_6_results(row['Team'], all_actuals_dfs)
        form_data.append(team_results)
    
    # Create color mapping
    def result_to_color(result):
        if result == 'W':
            return '#2ecc71'  # Green
        elif result == 'D':
            return '#f39c12'  # Orange
        elif result == 'L':
            return '#e74c3c'  # Red
        else:
            return '#ecf0f1'  # Light gray for no match
    
    # Create the figure
    fig = go.Figure()
    
    # Column headers
    headers = ['Pos', 'Team', 'MP', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts']
    form_headers = ['', '', '', '', '', '']  # For the 6 form columns
    
    # Prepare table data
    table_data = [
        latest_gw['Pos'].tolist(),
        latest_gw['Team'].tolist(),
        latest_gw['Pl'].tolist(),
        latest_gw['W'].tolist(),
        latest_gw['D'].tolist(),
        latest_gw['L'].tolist(),
        latest_gw['GF'].tolist(),
        latest_gw['GA'].tolist(),
        latest_gw['GD'].tolist(),
        latest_gw['Pts'].tolist(),
    ]
    
    # Add form columns (last 6 matches)
    for i in range(6):
        form_column = [match[i] if i < len(match) else '' for match in form_data]
        table_data.append(form_column)
        form_headers.append(f'')
    
    # Create cell colors for the form columns
    cell_colors = [['white'] * len(latest_gw) for _ in range(10)]  # Stats columns
    
    for i in range(6):
        form_column = [match[i] if i < len(match) else '' for match in form_data]
        colors = [result_to_color(result) for result in form_column]
        cell_colors.append(colors)
    
    # Create the table
    fig.add_trace(go.Table(
        header=dict(
            values=headers + ['Last 6 Matches →'] + [''] * 5,
            fill_color='#34495e',
            align=['center'] * 10 + ['center'] * 6,
            font=dict(color='white', size=12, family='Arial Black'),
            height=40
        ),
        cells=dict(
            values=table_data,
            fill_color=[['white'] * len(latest_gw) for _ in range(10)] + 
                       [[result_to_color(match[i]) if i < len(match) else '#ecf0f1' 
                         for match in form_data] for i in range(6)],
            align=['center'] * 10 + ['center'] * 6,
            font=dict(
                color=[['black'] * len(latest_gw) for _ in range(10)] + 
                      [['white'] * len(latest_gw) for _ in range(6)],
                size=11
            ),
            height=35
        ),
        columnwidth=[40, 150, 40, 40, 40, 40, 40, 40, 40, 50] + [35] * 6
    ))
    
    fig.update_layout(
        title=dict(
            text='Premier League Table with Recent Form',
            font=dict(size=18, family='Arial Black')
        ),
        height=max(600, len(latest_gw) * 40 + 100),
        margin=dict(l=10, r=10, t=60, b=10)
    )
    
    return fig

# Helper functions
def normalize_team_name(name):
    """Normalize team names to handle variations."""
    name_map = {
        "Tottenham Hotspur": "Tottenham",
        "Spurs": "Tottenham",
        "Manchester United": "Manchester Utd",
        "Man Utd": "Manchester Utd",
        "Man United": "Manchester Utd",
        "Manchester City": "Manchester City",
        "Newcastle United": "Newcastle Utd",
        "Nottingham Forest": "Nott'ham Forest",
        "Nott'm Forest": "Nott'ham Forest",
        "AFC Bournemouth": "Bournemouth",
        "West Ham United": "West Ham",
        "Wolverhampton Wanderers": "Wolves",
        "Brighton & Hove Albion": "Brighton"
        # Add other variations as needed
    }
    return name_map.get(name, name)

# Update the get_team_last_6_results function to use normalization:
def get_team_last_6_results(team_name, all_actuals_dfs):
    """Get last 6 results with team name normalization."""
    #normalized_name = normalize_team_name(team_name)
    ##print("==========================")
    ##print("==========================")
    ##print("==========================")
    ##print(f"get_team_last_6_results (185): STARTING ACTUALS LIST --> {all_actuals_dfs}")

    results = []
    
    for gw_df in all_actuals_dfs:
        ##print(f"get_team_last_6_results (188): Looping through GW DF from actuals --> {gw_df}")
        # Normalize team names in the dataframe
        gw_df_normalized = gw_df.copy()
        #gw_df_normalized['Home'] = gw_df_normalized['Home'].apply(normalize_team_name)
        #gw_df_normalized['Away'] = gw_df_normalized['Away'].apply(normalize_team_name)
        
        # Check home matches
        #home_matches = gw_df_normalized[gw_df_normalized['Home'] == normalized_name]
        home_matches = gw_df_normalized[gw_df_normalized['Home'] == team_name]

        ##print('=========')
        ##print(team_name)
        ##print("HOME Matches")
        ##print('=========')
        ##print(f"get_team_last_6_results(202): HOME MATCHES --> {home_matches}")
        
        i= 1
        
        for _, match in home_matches.iterrows():
            #print(f"get_team_last_6_results (207): HOME MATCH RESULT for match {i}--> {match['Result']}")
            if match['Result'] == 'Home Win':
                #print("Apending W")
                results.append('W')
            elif match['Result'] == 'Away Win':
                #print("Apending L")         
                results.append('L')
            elif match['Result'] == 'Tie':
                #print("Apending D")
                results.append('D')
                
            if len(results) >= 7 :
                break
            
            #print(f"get_team_last_6_results (221): LENGTH OF RESULTS --> {len(results)}")
            
            i += 1
            
        if len(results) >= 7:
            break
            
        # Check away matches
        #away_matches = gw_df_normalized[gw_df_normalized['Away'] == normalized_name]
        away_matches = gw_df_normalized[gw_df_normalized['Away'] == team_name]

        #print('=========')
        #print(team_name)
        #print("AWAY Matches")
        #print('=========')
        #print(f"get_team_last_6_results (236): AWAY MATCHES (away) --> {away_matches}")
        
        i= 1
        
        for _, match in away_matches.iterrows():
            #print(f"get_team_last_6_results (241): AWAY MATCH RESULT for match {i}--> {match['Result']}")
            if match['Result'] == 'Home Win':
                #print("Apending L")
                results.append('L')
            elif match['Result'] == 'Away Win':
                #print("Apending W")
                results.append('W')
            elif match['Result'] == 'Tie':
                #print("Apending D")
                results.append('D')
            
            if len(results) >= 7:
                break
            
            #print(f"get_team_last_6_results (255): LENGTH OF RESULTS (away) --> {len(results)}")
            
            i += 1
            
        if len(results) >= 7:
            break
    # #print(f"get_team_last_6_results (266): FINAL RESULTS BEFORE SLICED --> {results}")    
    # results = results[:6][::-2]
    # #print(f"get_team_last_6_results (266): FINAL RESULTS AFTER SLICED --> {results}")
    
    # while len(results) < 6  :
    #     results.insert(0, '')
    
    #print(f"get_team_last_6_results (266): FINAL RESULTS AFTER INSERT(0, '') for {team_name} --> {results}")
    
    return results

# Usage example:
# Combine all your game week dataframes into a list
# all_actuals = [gw_1_actuals, gw_2_actuals, gw_3_actuals, gw_4_actuals, 
#                gw_5_actuals, gw_6_actuals, gw_7_actuals]

# Create the visualization
# fig = create_team_form_table(table_all_df, all_actuals)
# st.plotly_chart(fig, use_container_width=True)

###### Enhanced Visual ##########

def create_enhanced_team_form_table(table_df, all_actuals_dfs):
    """
    Enhanced version with better styling and tooltips.
    """
    
    # Get the latest game week data
    latest_gw = table_df[table_df['Pl'] == table_df['Pl'].max()].copy()
    latest_gw = latest_gw.sort_values('Pos').reset_index(drop=True)
    
    #print(f"create_enhanced_team_form_table (290): LATEST GW --> {latest_gw}")
    
    # Get last 6 results for each team
    form_data = []
    for _, row in latest_gw.iterrows():
        team_results = get_team_last_6_results(row['Team'], all_actuals_dfs)
        form_data.append(team_results)
        
    #print(f"create_enhanced_team_form_table (296): FORM DATA --> {form_data}")
    
    # Calculate form points (last 6 matches)
    form_points = []
    for results in form_data:
        points = sum(3 if r == 'W' else 1 if r == 'D' else 0 for r in results if r)
        form_points.append(points)
    
    #print(f"create_enhanced_team_form_table (302): FORM POINTS --> {form_points}")
    
    # Create the figure with subplots approach for better control
    fig = go.Figure()
    
    # Prepare all data
    headers = ['Pos', 'Team', 'MP', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts', 'Form']
    
    # Add match labels (GW-6, GW-5, etc.)
    current_gw = int(latest_gw['Pl'].iloc[0])
    match_labels = [f'GW{current_gw-5+i}' if current_gw-5+i > 0 else '' for i in range(6)]
    headers.extend(match_labels)
    
    ##print(f"create_enhanced_team_form_table (315): MATCH LABELS --> {match_labels}")
    
    # Table data
    table_data = [
        latest_gw['Pos'].tolist(),
        latest_gw['Team'].tolist(),
        latest_gw['Pl'].tolist(),
        latest_gw['W'].tolist(),
        latest_gw['D'].tolist(),
        latest_gw['L'].tolist(),
        latest_gw['GF'].tolist(),
        latest_gw['GA'].tolist(),
        latest_gw['GD'].tolist(),
        latest_gw['Pts'].tolist(),
        form_points
    ]
    
    # Add form match results
    for i in range(7):
        if i > 0:
            form_column = [match[i] if i < len(match) else '' for match in form_data]
            table_data.append(form_column)
    
    ##print(f"create_enhanced_team_form_table (337): TABLE DATA --> {table_data}")
    
    # Color coding
    def result_to_color(result):
        colors = {
            'W': '#27ae60',  # Green
            'D': '#f39c12',  # Orange
            'L': '#e74c3c',  # Red
            '': '#ecf0f1'    # Light gray
        }
        return colors.get(result, '#ecf0f1')
    
    # Create cell colors
    # First 10 columns: alternate row colors for readability
    base_colors = []
    for i in range(len(latest_gw)):
        color = '#f8f9fa' if i % 2 == 0 else 'white'
        base_colors.append(color)
    
    cell_fill = [base_colors for _ in range(11)]  # 11 stat columns
    
    # Form columns: colored by result
    for i in range(7):
        if i > 0:
            form_column = [match[i] if i < len(match) else '' for match in form_data]
            colors = [result_to_color(result) for result in form_column]
            cell_fill.append(colors)
    
    # Font colors
    font_colors = [['black'] * len(latest_gw) for _ in range(11)]
    font_colors.extend([['white'] * len(latest_gw) for _ in range(6)])
    
    # Create table
    fig.add_trace(go.Table(
        header=dict(
            values=headers,
            fill_color='#2c3e50',
            align=['center'] * len(headers),
            font=dict(color='white', size=11, family='Arial'),
            height=35
        ),
        cells=dict(
            values=table_data,
            fill_color=cell_fill,
            align=['center'] * len(headers),
            font=dict(
                color=font_colors,
                size=10,
                family='Arial'
            ),
            height=30
        ),
        columnwidth=[35, 150, 35, 35, 35, 35, 40, 40, 40, 45, 45] + [35] * 6
    ))
    
    fig.update_layout(
        title=dict(
            text=f'Premier League Table - Game Week {current_gw} (with Last 6 Matches)',
            font=dict(size=16, family='Arial', color='#2c3e50'),
            x=0.5,
            xanchor='center'
        ),
        height=max(550, len(latest_gw) * 32 + 120),
        margin=dict(l=10, r=10, t=70, b=10),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    return fig
