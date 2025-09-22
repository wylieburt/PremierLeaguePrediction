import matplotlib.pyplot as plt


def team_performance(df):
    # Group by team and sum W, D, L values (in case there are multiple rows per team)
    team_stats = df.groupby('Team')[['W', 'D', 'L']].first().reset_index()
    team_stats = team_stats.sort_values('W', ascending=False)
    
    # Create the stacked horizontal bar chart
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create the bars
    bars_w = ax.barh(team_stats['Team'], team_stats['W'], 
                     label='Wins', color='#2E8B57', alpha=0.8)
    bars_d = ax.barh(team_stats['Team'], team_stats['D'], 
                     left=team_stats['W'], label='Draws', color='#FFD700', alpha=0.8)
    bars_l = ax.barh(team_stats['Team'], team_stats['L'], 
                     left=team_stats['W'] + team_stats['D'], 
                     label='Losses', color='#DC143C', alpha=0.8)
    
    # Customize the chart
    ax.set_xlabel('Number of Games', fontsize=12, fontweight='bold')
    ax.set_title('Team Performance: Wins, Draws, and Losses', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    
    # Add grid for better readability
    ax.grid(axis='x', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add value labels on the bars
    for i, team in enumerate(team_stats['Team']):
        w_val = team_stats.iloc[i]['W']
        d_val = team_stats.iloc[i]['D']
        l_val = team_stats.iloc[i]['L']
        
        # Add win count
        if w_val > 5:  # Only show if bar is wide enough
            ax.text(w_val/2, i, str(w_val), ha='center', va='center', 
                    fontweight='bold', color='white')
        
        # Add draw count
        if d_val > 5:  # Only show if bar is wide enough
            ax.text(w_val + d_val/2, i, str(d_val), ha='center', va='center', 
                    fontweight='bold', color='black')
        
        # Add loss count
        if l_val > 5:  # Only show if bar is wide enough
            ax.text(w_val + d_val + l_val/2, i, str(l_val), ha='center', va='center', 
                    fontweight='bold', color='white')
    
    plt.tight_layout()   

    return fig