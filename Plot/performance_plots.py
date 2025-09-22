import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

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

def preprocess_football_data(df):
    """
    Preprocess English football match data
    Expected columns: Date (string), Result (W/L/D), count (number of matches)
    """
    processed_df = df.copy()
    
    # Convert Date string to datetime
    processed_df['Date'] = pd.to_datetime(processed_df['Date'])
    
    # Extract date components for football season analysis
    processed_df['Year'] = processed_df['Date'].dt.year
    processed_df['Month'] = processed_df['Date'].dt.month
    processed_df['DayOfWeek'] = processed_df['Date'].dt.dayofweek
    
    # Football season logic (Aug-May)
    # Season starts in August and ends in May of the following year
    processed_df['Season'] = processed_df['Date'].apply(get_football_season)
    
    print("Football data preprocessing complete!")
    print(f"Date range: {processed_df['Date'].min().strftime('%Y-%m-%d')} to {processed_df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Seasons covered: {sorted(processed_df['Season'].unique())}")
    
    return processed_df

def get_football_season(date):
    """Convert date to football season (e.g., 2023-24 season)"""
    if date.month >= 8:  # August onwards is new season
        return f"{date.year}-{str(date.year + 1)[2:]}"
    else:  # January-July is previous season
        return f"{date.year - 1}-{str(date.year)[2:]}"

def create_football_performance_dashboard(df):
    """
    Create comprehensive dashboard for English football performance analysis
    """
    # Process daily results
    daily_results = df.groupby(['Date', 'Result'])['count'].sum().unstack(fill_value=0)
    
    # Ensure all result types exist
    for col in ['W', 'L', 'D']:
        if col not in daily_results.columns:
            daily_results[col] = 0
    
    # Calculate cumulative and rolling metrics
    daily_results['Total'] = daily_results[['W', 'L', 'D']].sum(axis=1)
    daily_results['Win_Rate'] = daily_results['W'] / daily_results['Total']
    daily_results['Points'] = daily_results['W'] * 3 + daily_results['D'] * 1  # Football points system
    daily_results['Points_Per_Game'] = daily_results['Points'] / daily_results['Total']
    
    # Rolling averages (useful for form analysis)
    daily_results['Rolling_Win_Rate_5'] = daily_results['Win_Rate'].rolling(window=5, min_periods=1).mean()
    daily_results['Rolling_Win_Rate_10'] = daily_results['Win_Rate'].rolling(window=10, min_periods=1).mean()
    daily_results['Rolling_PPG_5'] = daily_results['Points_Per_Game'].rolling(window=5, min_periods=1).mean()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Stacked Area Chart - Match Results Over Time
    ax1 = plt.subplot(3, 2, 1)
    create_stacked_area_chart(daily_results, ax1)
    
    # 2. Multiple Line Chart - Win/Loss/Draw Trends
    ax2 = plt.subplot(3, 2, 2)
    create_multiple_line_chart(daily_results, ax2)
    
    # 3. Rolling Win Percentage
    ax3 = plt.subplot(3, 2, 3)
    create_rolling_win_percentage(daily_results, ax3)
    
    # 4. Points Per Game Analysis
    ax4 = plt.subplot(3, 2, 4)
    create_points_analysis(daily_results, ax4)
    
    # 5. Seasonal Performance Comparison
    ax5 = plt.subplot(3, 2, 5)
    create_seasonal_analysis(df, ax5)
    
    # 6. Form Guide (Recent Performance)
    ax6 = plt.subplot(3, 2, 6)
    create_form_guide(daily_results, ax6)
    
    plt.suptitle('English Football Performance Dashboard', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig
    
    # Print summary statistics
    #print_performance_summary(daily_results, df)

def create_stacked_area_chart(daily_results, ax):
    """Create stacked area chart showing match results over time"""
    # Resample to weekly data for cleaner visualization
    weekly_data = daily_results.resample('W')[['W', 'L', 'D']].sum()
    
    # Create stacked area chart
    ax.stackplot(weekly_data.index, 
                weekly_data['W'], weekly_data['D'], weekly_data['L'],
                labels=['Wins', 'Draws', 'Losses'],
                colors=['#2ecc71', '#f39c12', '#e74c3c'],
                alpha=0.8)
    
    ax.set_title('Match Results Over Time (Stacked Area)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Matches')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.tick_params(axis='x', rotation=45)

def create_multiple_line_chart(daily_results, ax):
    """Create multiple line chart showing win/loss/draw trends"""
    # Resample to monthly data and calculate percentages
    monthly_data = daily_results.resample('M')[['W', 'L', 'D', 'Total']].sum()
    monthly_data['Win_Pct'] = monthly_data['W'] / monthly_data['Total'] * 100
    monthly_data['Draw_Pct'] = monthly_data['D'] / monthly_data['Total'] * 100
    monthly_data['Loss_Pct'] = monthly_data['L'] / monthly_data['Total'] * 100
    
    # Plot lines
    ax.plot(monthly_data.index, monthly_data['Win_Pct'], 
           marker='o', linewidth=2.5, label='Win %', color='#2ecc71')
    ax.plot(monthly_data.index, monthly_data['Draw_Pct'], 
           marker='s', linewidth=2.5, label='Draw %', color='#f39c12')
    ax.plot(monthly_data.index, monthly_data['Loss_Pct'], 
           marker='^', linewidth=2.5, label='Loss %', color='#e74c3c')
    
    ax.set_title('Win/Draw/Loss Percentage Trends (Monthly)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Date')
    ax.set_ylabel('Percentage (%)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    # Format x-axis
    ax.tick_params(axis='x', rotation=45)

def create_rolling_win_percentage(daily_results, ax):
    """Create rolling win percentage chart with multiple windows"""
    # Calculate rolling averages with different windows
    daily_results['Rolling_3'] = daily_results['Win_Rate'].rolling(window=3, min_periods=1).mean() * 100
    daily_results['Rolling_5'] = daily_results['Rolling_Win_Rate_5'] * 100
    daily_results['Rolling_10'] = daily_results['Rolling_Win_Rate_10'] * 100
    
    # Plot rolling averages
    ax.plot(daily_results.index, daily_results['Rolling_3'], 
           alpha=0.7, linewidth=1.5, label='3-match form', color='#3498db')
    ax.plot(daily_results.index, daily_results['Rolling_5'], 
           linewidth=2.5, label='5-match form', color='#9b59b6')
    ax.plot(daily_results.index, daily_results['Rolling_10'], 
           linewidth=3, label='10-match form', color='#e67e22')
    
    # Add overall win rate as horizontal line
    overall_win_rate = (daily_results['W'].sum() / daily_results['Total'].sum()) * 100
    ax.axhline(y=overall_win_rate, color='red', linestyle='--', alpha=0.7, 
              label=f'Overall Win Rate ({overall_win_rate:.1f}%)')
    
    ax.set_title('Rolling Win Percentage (Form Guide)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Date')
    ax.set_ylabel('Win Percentage (%)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    # Format x-axis
    ax.tick_params(axis='x', rotation=45)

def create_points_analysis(daily_results, ax):
    """Create points per game analysis (3 for win, 1 for draw)"""
    # Calculate cumulative points and PPG
    daily_results['Cumulative_Points'] = daily_results['Points'].cumsum()
    daily_results['Cumulative_Games'] = daily_results['Total'].cumsum()
    daily_results['Cumulative_PPG'] = daily_results['Cumulative_Points'] / daily_results['Cumulative_Games']
    
    # Plot PPG trends
    ax.plot(daily_results.index, daily_results['Rolling_PPG_5'], 
           linewidth=2.5, label='5-match PPG', color='#16a085')
    ax.plot(daily_results.index, daily_results['Cumulative_PPG'], 
           linewidth=2, label='Season PPG', color='#8e44ad', alpha=0.8)
    
    # Add reference lines for different performance levels
    ax.axhline(y=2.0, color='green', linestyle=':', alpha=0.7, label='Title Form (2.0+ PPG)')
    ax.axhline(y=1.5, color='orange', linestyle=':', alpha=0.7, label='Top 6 Form (1.5+ PPG)')
    ax.axhline(y=1.0, color='red', linestyle=':', alpha=0.7, label='Relegation Form (1.0 PPG)')
    
    ax.set_title('Points Per Game Analysis', fontweight='bold', fontsize=12)
    ax.set_xlabel('Date')
    ax.set_ylabel('Points Per Game')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 3)
    
    # Format x-axis
    ax.tick_params(axis='x', rotation=45)

def create_seasonal_analysis(df, ax):
    """Compare performance across different seasons"""
    # Add season column for grouping
    df_with_season = df.copy()
    df_with_season['Season'] = df_with_season['Date'].apply(get_football_season)
    
    # Calculate seasonal statistics
    seasonal_stats = df_with_season.groupby(['Season', 'Result'])['count'].sum().unstack(fill_value=0)
    
    # Ensure all columns exist
    for col in ['W', 'L', 'D']:
        if col not in seasonal_stats.columns:
            seasonal_stats[col] = 0
    
    seasonal_stats['Total'] = seasonal_stats[['W', 'L', 'D']].sum(axis=1)
    seasonal_stats['Win_Rate'] = seasonal_stats['W'] / seasonal_stats['Total'] * 100
    seasonal_stats['PPG'] = (seasonal_stats['W'] * 3 + seasonal_stats['D']) / seasonal_stats['Total']
    
    # Create bar chart
    x_pos = range(len(seasonal_stats))
    
    bars1 = ax.bar(x_pos, seasonal_stats['Win_Rate'], alpha=0.8, 
                  color='#2ecc71', label='Win Rate %')
    
    # Add PPG on secondary y-axis
    ax2 = ax.twinx()
    line = ax2.plot(x_pos, seasonal_stats['PPG'], 'ro-', linewidth=2, 
                   markersize=6, label='Points Per Game')
    
    ax.set_title('Seasonal Performance Comparison', fontweight='bold', fontsize=12)
    ax.set_xlabel('Season')
    ax.set_ylabel('Win Rate (%)', color='#2ecc71')
    ax2.set_ylabel('Points Per Game', color='red')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(seasonal_stats.index, rotation=45)
    
    # Add value labels on bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

def create_form_guide(daily_results, ax):
    """Create recent form guide visualization"""
    # Get last 20 match days for form analysis
    recent_form = daily_results.tail(20).copy()
    
    if len(recent_form) == 0:
        ax.text(0.5, 0.5, 'No recent data available', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Create color-coded form chart
    colors = []
    results_text = []
    
    for idx, row in recent_form.iterrows():
        if row['W'] > row['L'] and row['W'] > row['D']:
            colors.append('#2ecc71')  # Green for win
            results_text.append('W')
        elif row['L'] > row['W'] and row['L'] > row['D']:
            colors.append('#e74c3c')  # Red for loss
            results_text.append('L')
        else:
            colors.append('#f39c12')  # Orange for draw
            results_text.append('D')
    
    # Create form chart
    x_pos = range(len(recent_form))
    bars = ax.bar(x_pos, [1] * len(recent_form), color=colors, alpha=0.8)
    
    # Add result letters on bars
    for i, (bar, result) in enumerate(zip(bars, results_text)):
        ax.text(bar.get_x() + bar.get_width()/2., 0.5, result,
               ha='center', va='center', fontweight='bold', color='white', fontsize=12)
    
    ax.set_title('Recent Form Guide (Last 20 Match Days)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Match Day (Recent →)')
    ax.set_ylabel('')
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    
    # Add form summary
    recent_wins = sum(1 for r in results_text if r == 'W')
    recent_draws = sum(1 for r in results_text if r == 'D')
    recent_losses = sum(1 for r in results_text if r == 'L')
    recent_points = recent_wins * 3 + recent_draws
    recent_ppg = recent_points / len(results_text) if len(results_text) > 0 else 0
    
    form_text = f"Form: W{recent_wins} D{recent_draws} L{recent_losses} | PPG: {recent_ppg:.2f}"
    ax.text(0.5, -0.15, form_text, ha='center', va='top', transform=ax.transAxes,
           fontsize=10, fontweight='bold')

def print_performance_summary(daily_results, df):
    """Print comprehensive performance summary"""
    #st.write("\n" + "="*60)
    #st.write("ENGLISH FOOTBALL PERFORMANCE SUMMARY")
    #st.write("="*60)
    
    # Overall statistics
    total_games = daily_results['Total'].sum()
    total_wins = daily_results['W'].sum()
    total_draws = daily_results['D'].sum()
    total_losses = daily_results['L'].sum()
    total_points = total_wins * 3 + total_draws
    
    win_rate = (total_wins / total_games) * 100 if total_games > 0 else 0
    ppg = total_points / total_games if total_games > 0 else 0
    
   # st.write(f"Overall Record: W {total_wins} D {total_draws} L {total_losses}")
   # st.write(f"Total Games: {total_games}")
   #st.write(f"Win Rate: {win_rate:.1f}%")
   # st.write(f"Points: {total_points}")
   # st.write(f"Points Per Game: {ppg:.2f}")
    
    # Performance categorization
    if ppg >= 2.0:
        performance = "Title-challenging form! 🏆"
    elif ppg >= 1.5:
        performance = "Top 6 form 📈"
    elif ppg >= 1.2:
        performance = "Mid-table form ⚽"
    elif ppg >= 1.0:
        performance = "Struggling form ⚠️"
    else:
        performance = "Relegation form! 🚨"
    
    print(f"Performance Level: {performance}")
    
    # Recent form (last 10 games)
    recent_10 = daily_results.tail(10)
    if len(recent_10) > 0:
        recent_wins = recent_10['W'].sum()
        recent_draws = recent_10['D'].sum()
        recent_losses = recent_10['L'].sum()
        recent_games = recent_10['Total'].sum()
        recent_points = recent_wins * 3 + recent_draws
        recent_ppg = recent_points / recent_games if recent_games > 0 else 0
        
        #st.write(f"\nRecent Form (Last {recent_games} games):")
        #st.write(f"Record: W {recent_wins} D {recent_draws} L {recent_losses}")
        #st.write(f"Recent PPG: {recent_ppg:.2f}")

# Sample data creation for English football
def create_sample_football_data():
    """Create realistic English football sample data"""
    np.random.seed(42)
    
    # Create data for multiple seasons
    start_date = datetime(2020, 8, 15)  # Start of 2020-21 season
    end_date = datetime(2025, 5, 15)    # End of 2024-25 season
    
    sample_data = []
    current_date = start_date
    
    while current_date <= end_date:
        # Football typically has matches on weekends and midweek
        if current_date.weekday() in [1, 5, 6]:  # Tuesday, Saturday, Sunday
            # Random match results with realistic distribution
            # Better teams have more wins, fewer losses
            match_results = np.random.choice(['W', 'D', 'L'], 
                                           size=np.random.randint(1, 4),
                                           p=[0.5, 0.3, 0.2])  # Decent team performance
            
            for result in match_results:
                count = np.random.randint(1, 3)  # Usually 1-2 matches per day
                sample_data.append({
                    'Date': current_date.strftime('%Y-%m-%d'),
                    'Result': result,
                    'count': count
                })
        
        current_date += timedelta(days=1)
    
    return pd.DataFrame(sample_data)

