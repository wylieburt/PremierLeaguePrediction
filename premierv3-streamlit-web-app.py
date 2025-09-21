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

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


#Plot functions for data tab


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
    st.pyplot(fig)
    
    # Print summary statistics
    print_performance_summary(daily_results, df)

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
    st.write("\n" + "="*60)
    st.write("ENGLISH FOOTBALL PERFORMANCE SUMMARY")
    st.write("="*60)
    
    # Overall statistics
    total_games = daily_results['Total'].sum()
    total_wins = daily_results['W'].sum()
    total_draws = daily_results['D'].sum()
    total_losses = daily_results['L'].sum()
    total_points = total_wins * 3 + total_draws
    
    win_rate = (total_wins / total_games) * 100 if total_games > 0 else 0
    ppg = total_points / total_games if total_games > 0 else 0
    
    st.write(f"Overall Record: W {total_wins} D {total_draws} L {total_losses}")
    st.write(f"Total Games: {total_games}")
    st.write(f"Win Rate: {win_rate:.1f}%")
    st.write(f"Points: {total_points}")
    st.write(f"Points Per Game: {ppg:.2f}")
    
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
        
        st.write(f"\nRecent Form (Last {recent_games} games):")
        st.write(f"Record: W {recent_wins} D {recent_draws} L {recent_losses}")
        st.write(f"Recent PPG: {recent_ppg:.2f}")

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



def create_heatmaps(df):
    # Process data: aggregate by date (Date is now datetime)
    daily_stats = df.groupby(['Date', 'Result'])['count'].sum().unstack(fill_value=0)
    
    # Handle missing columns gracefully
    for col in ['W', 'L', 'D']:
        if col not in daily_stats.columns:
            daily_stats[col] = 0
    
    # Calculate metrics
    daily_stats['Total'] = daily_stats[['W', 'L', 'D']].sum(axis=1)
    daily_stats['Win_Rate'] = daily_stats['W'] / daily_stats['Total'].replace(0, 1)
    
    # Add date features to daily_stats
    daily_stats['Year'] = daily_stats.index.year
    daily_stats['Month'] = daily_stats.index.month
    daily_stats['DayOfWeek'] = daily_stats.index.dayofweek
    daily_stats['Week'] = daily_stats.index.isocalendar().week
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Game Results Heatmap Analysis', fontsize=16, y=0.98)
    
    # 1. Calendar Heatmap - Win Rate
    ax1 = axes[0, 0]
    create_calendar_heatmap(daily_stats, ax1, 'Win_Rate', 'Win Rate', 'RdYlGn')
    
    # 2. Calendar Heatmap - Total Games
    ax2 = axes[0, 1]
    create_calendar_heatmap(daily_stats, ax2, 'Total', 'Total Games', 'Blues')
    
    # 3. Monthly Aggregation Heatmap
    ax3 = axes[1, 0]
    create_monthly_heatmap(daily_stats, ax3)
    
    # 4. Day of Week Pattern
    ax4 = axes[1, 1]
    create_weekday_heatmap(daily_stats, ax4)
    
    # 5. Weekly Rolling Average
    ax5 = axes[2, 0]
    create_rolling_heatmap(daily_stats, ax5)
    
    # 6. Yearly Comparison
    ax6 = axes[2, 1]
    create_yearly_comparison_heatmap(daily_stats, ax6)
    
    plt.tight_layout()
    st.pyplot(fig)

def create_calendar_heatmap(daily_stats, ax, metric_col, title, cmap):
    """Create a calendar-style heatmap using date features"""
    # Get the most recent complete year with data
    available_years = daily_stats['Year'].unique()
    target_year = max(available_years) if len(available_years) > 0 else 2024
    
    # Filter for specific year
    year_data = daily_stats[daily_stats['Year'] == target_year]
    
    if len(year_data) == 0:
        ax.text(0.5, 0.5, f'No data for {target_year}', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{title} - {target_year}')
        return
    
    # Create matrix for calendar (53 weeks x 7 days)
    calendar_matrix = np.full((53, 7), np.nan)
    
    for idx, row in year_data.iterrows():
        week = min(row['Week'] - 1, 52)  # Week of year (0-indexed, capped at 52)
        day = row['DayOfWeek']  # Monday=0, Sunday=6
        calendar_matrix[week, day] = row[metric_col]
    
    # Create heatmap
    im = ax.imshow(calendar_matrix.T, cmap=cmap, aspect='auto', interpolation='nearest')
    ax.set_title(f'{title} - {target_year} Calendar View')
    ax.set_xlabel('Week of Year')
    ax.set_ylabel('Day of Week')
    ax.set_yticks(range(7))
    ax.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    
    # Set x-axis to show every 4th week
    ax.set_xticks(range(0, 53, 4))
    ax.set_xticklabels(range(1, 54, 4))
    
    plt.colorbar(im, ax=ax, shrink=0.6)

def create_monthly_heatmap(daily_stats, ax):
    """Create monthly aggregation heatmap using Year and Month features"""
    # Group by year and month using the extracted features
    monthly = daily_stats.groupby(['Year', 'Month']).agg({
        'W': 'sum', 'L': 'sum', 'D': 'sum', 'Total': 'sum'
    })
    monthly['Win_Rate'] = monthly['W'] / monthly['Total'].replace(0, 1)
    
    # Create pivot table for heatmap
    monthly_pivot = monthly['Win_Rate'].unstack(level=0, fill_value=0)
    
    if not monthly_pivot.empty:
        sns.heatmap(monthly_pivot, annot=True, fmt='.2f', cmap='RdYlGn', 
                   ax=ax, cbar_kws={'shrink': 0.6}, center=0.5)
        ax.set_title('Monthly Win Rate Heatmap')
        ax.set_xlabel('Year')
        ax.set_ylabel('Month')
    else:
        ax.text(0.5, 0.5, 'Insufficient data for monthly view', ha='center', va='center', transform=ax.transAxes)

def create_weekday_heatmap(daily_stats, ax):
    """Create day of week pattern heatmap using DayOfWeek and Month features"""
    # Group by month and weekday using extracted features
    weekday_monthly = daily_stats.groupby(['Month', 'DayOfWeek'])['Win_Rate'].mean().unstack(fill_value=0)
    
    if not weekday_monthly.empty and weekday_monthly.shape[0] > 1:
        sns.heatmap(weekday_monthly, annot=True, fmt='.2f', cmap='viridis',
                   ax=ax, cbar_kws={'shrink': 0.6})
        ax.set_title('Win Rate by Day of Week and Month')
        ax.set_xlabel('Day of Week (0=Monday, 6=Sunday)')
        ax.set_ylabel('Month')
    else:
        ax.text(0.5, 0.5, 'Insufficient data for weekday analysis', ha='center', va='center', transform=ax.transAxes)

def create_rolling_heatmap(daily_stats, ax):
    """Create rolling average heatmap"""
    # Calculate 7-day rolling average
    rolling_winrate = daily_stats['Win_Rate'].rolling(window=7, center=True).mean()
    rolling_total = daily_stats['Total'].rolling(window=7, center=True).mean()
    
    # Sample every 7 days for visualization
    sample_indices = range(0, len(rolling_winrate), 7)
    sample_winrates = rolling_winrate.iloc[sample_indices]
    sample_totals = rolling_total.iloc[sample_indices]
    
    # Remove NaN values
    valid_mask = ~(sample_winrates.isna() | sample_totals.isna())
    sample_winrates = sample_winrates[valid_mask]
    sample_totals = sample_totals[valid_mask]
    
    if len(sample_winrates) > 0:
        # Create scatter plot with color coding
        scatter = ax.scatter(range(len(sample_winrates)), sample_winrates, 
                            c=sample_totals, cmap='plasma', s=60, alpha=0.7)
        ax.plot(range(len(sample_winrates)), sample_winrates, alpha=0.3, color='gray')
        
        ax.set_title('7-Day Rolling Win Rate (Color = Game Volume)')
        ax.set_xlabel('Time Period (Weekly Samples)')
        ax.set_ylabel('Rolling Win Rate')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, shrink=0.6, label='Avg Games/Day')
    else:
        ax.text(0.5, 0.5, 'Insufficient data for rolling analysis', ha='center', va='center', transform=ax.transAxes)

def create_yearly_comparison_heatmap(daily_stats, ax):
    """Create yearly comparison heatmap using Year and Month features"""
    # Group by year and month for comparison
    yearly_monthly = daily_stats.groupby(['Year', 'Month']).agg({
        'Win_Rate': 'mean',
        'Total': 'sum'
    })
    
    # Create pivot for win rate by month across years
    yearly_pivot = yearly_monthly['Win_Rate'].unstack(level=0, fill_value=np.nan)
    
    if not yearly_pivot.empty and yearly_pivot.shape[1] > 1:
        sns.heatmap(yearly_pivot, annot=True, fmt='.2f', cmap='RdYlGn',
                   ax=ax, cbar_kws={'shrink': 0.6}, center=0.5)
        ax.set_title('Win Rate by Month Across Years')
        ax.set_xlabel('Year')
        ax.set_ylabel('Month')
    else:
        ax.text(0.5, 0.5, 'Multiple years needed for comparison', ha='center', va='center', transform=ax.transAxes)

# Alternative: Simple focused heatmap functions using date features
def simple_win_rate_heatmap(df, year=None):
    """Create a simple, focused win rate calendar heatmap using date features"""
    # Process data
    daily_stats = df.groupby(['Date', 'Result'])['count'].sum().unstack(fill_value=0)
    
    # Handle missing columns
    for col in ['W', 'L', 'D']:
        if col not in daily_stats.columns:
            daily_stats[col] = 0
            
    daily_stats['Total'] = daily_stats[['W', 'L', 'D']].sum(axis=1)
    daily_stats['Win_Rate'] = daily_stats['W'] / daily_stats['Total'].replace(0, 1)
    
    # Add date features
    daily_stats['Year'] = daily_stats.index.year
    daily_stats['Week'] = daily_stats.index.isocalendar().week
    daily_stats['DayOfWeek'] = daily_stats.index.dayofweek
    
    # Auto-select year if not provided
    if year is None:
        available_years = daily_stats['Year'].unique()
        year = max(available_years) if len(available_years) > 0 else 2024
    
    # Filter for specific year
    year_data = daily_stats[daily_stats['Year'] == year]
    
    if len(year_data) == 0:
        print(f"No data available for year {year}")
        return
    
    # Create calendar matrix (53 weeks x 7 days)
    calendar_data = np.full((53, 7), np.nan)
    
    # Fill calendar matrix using date features
    for idx, row in year_data.iterrows():
        week = min(row['Week'] - 1, 52)  # 0-indexed, capped at 52
        day = row['DayOfWeek']  # Monday=0, Sunday=6
        calendar_data[week, day] = row['Win_Rate']
    
    # Plot
    plt.figure(figsize=(15, 8))
    
    im = plt.imshow(calendar_data.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    plt.colorbar(im, label='Win Rate', shrink=0.8)
    plt.title(f'Win Rate Calendar Heatmap - {year}', fontsize=14, fontweight='bold')
    plt.xlabel('Week of Year')
    plt.ylabel('Day of Week')
    plt.yticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    
    # Set x-axis to show every 4th week
    plt.xticks(range(0, 53, 4), range(1, 54, 4))
    
    # Add month boundaries (approximate)
    month_weeks = [1, 5, 9, 14, 18, 22, 27, 31, 35, 40, 44, 48]  # Approximate week numbers for month starts
    for week in month_weeks:
        if week < 53:
            plt.axvline(x=week-1, color='white', linewidth=1, alpha=0.7)
    
    plt.tight_layout()
    st.pyplot(plt)
    
    # Print summary stats
    st.write(f"\n{year} Summary:")
    st.write(f"Total games: {year_data['Total'].sum()}")
    st.write(f"Win rate: {year_data['W'].sum() / year_data['Total'].sum():.2%}")
    st.write(f"Days with games: {len(year_data)}")

# Monthly summary with better date feature usage
def monthly_summary_heatmap(df):
    """Create a focused monthly summary heatmap using date features"""
    # Group by year and month using the preprocessed features
    monthly = df.groupby([df['Year'], df['Month'], df['Result']])['count'].sum().unstack(fill_value=0)
    
    # Handle missing columns
    for col in ['W', 'L', 'D']:
        if col not in monthly.columns:
            monthly[col] = 0
    
    monthly['Total'] = monthly[['W', 'L', 'D']].sum(axis=1)
    monthly['Win_Rate'] = monthly['W'] / monthly['Total'].replace(0, 1)
    
    # Reset index to work with year/month as columns
    monthly_reset = monthly.reset_index()
    
    # Create pivot table
    pivot_wr = monthly_reset.pivot(index='Month', columns='Year', values='Win_Rate')
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_wr, annot=True, fmt='.2f', cmap='RdYlGn', 
                center=0.5, vmin=0, vmax=1, cbar_kws={'label': 'Win Rate'})
    plt.title('Monthly Win Rate Heatmap by Year', fontweight='bold', fontsize=14)
    plt.ylabel('Month')
    plt.xlabel('Year')
    
    # Add month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.yticks(range(len(pivot_wr.index)), [month_names[i-1] for i in pivot_wr.index])
    
    plt.tight_layout()
    st.pyplot(plt)  

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
        return ['background-color: #08f10a'] * len(row)  # Green
    elif row['Result'] != row['Predicted']:
        return ['background-color: #f13c08'] * len(row)  # Red
    else:
        return [''] * len(row)
    
def enhanced_results_summary(df):
    """
    Create enhanced summary with totals and percentages
    """
    # Basic summary
    result_summary = df.groupby('Team')['Result'].value_counts().unstack(fill_value=0)
    
    # Ensure all columns exist
    for col in ['W', 'D', 'L']:
        if col not in result_summary.columns:
            result_summary[col] = 0
    
    # Add calculated columns
    result_summary['Total_Games'] = result_summary['W'] + result_summary['D'] + result_summary['L']
    result_summary['Win_Pct'] = (result_summary['W'] / result_summary['Total_Games'] * 100).round(1)
    result_summary['Points'] = result_summary['W'] * 3 + result_summary['D'] * 1  # Football points system
    
    # Select columns and reset index
    result_summary = result_summary[['W', 'D', 'L', 'Total_Games', 'Win_Pct', 'Points']].reset_index()
    
    # Sort by Points (highest to lowest)
    result_summary = result_summary.sort_values('Points', ascending=False).reset_index(drop=True)
    
    return result_summary

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




#timeseries_df = preprocess_timeseries_data(timeseries_df)


columns_to_keep = ['Date','Team', 'Opp', 'Result']
match_result_lookup = all_data_df[columns_to_keep]

enhanced_results_summary_df = enhanced_results_summary(match_result_lookup)

merged_df = pd.merge(data_for_avg, enhanced_results_summary_df, how = "inner", on = "Team")

accuracy_tracking = pd.DataFrame({"Game Week" : ["GW 1", "GW 2", "GW 3", "GW 4", "GW 5"],
                                  "Accuracy" : [60, 70, 40, 70, 50],
                                  "Running Median" : [60, 65, 60, 65, 60]})


################################################
# Create tabs to display at top of each tab
################################################

tab1, tab2, tab3 = st.tabs(["Make Prediction", "Data and Model", "About Project"])

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
            
            #print(abs(len(tmp_team1) - len(tmp_team2))

            # if one team has played less matches than the other apply a Baysian Shrinkage
            # to make sure the averages for that team are reliable.  This is more pronounced
            # for the teams that are newer to the premier league.
            
            if (len(tmp_team1) < 50) | (len(tmp_team2) < 50):
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
            
            #col1.markdown(f"""
            #    <span style="color: black;">Our Prediction Using: {clf_reduced_name}</span><br><br>
            #    """, unsafe_allow_html=True) 
            
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
    
    # Every form must have a submit button.
    #submitted = st.form_submit_button("See Matches")
    #if submitted:
    

    
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
    
    # gw_num_actuals = left.selectbox(
    #     "Pick a game week:",
    #     ("game_week1",
    #      "game_week2",
    #      "game_week3",
    #      "game_week4",
    #      "game_week5"
    #      ),  key="gw_num_actuls")
    
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
            styled_df = selected_dataframe.style.apply(highlight_multiple_conditions, axis=1)        
            st.dataframe(styled_df, column_order= ("Date", "Home","Away", "Score", "Predicted"), hide_index=True)
            st.text(f"Predicted correct accuracy: {num_right} of {total} played - {accuracy:.0%}")
        else:
            styled_df = selected_dataframe.style.apply(highlight_multiple_conditions, axis=1)        
            st.dataframe(styled_df, column_order= ("Date","Home","Away", "Score", "Predicted"), hide_index=True)
        
    else:
        st.write(f"DataFrame for {gw_num_pick} not yet implemented")
        st.text("Predicted correct accuracy: NONE")
        
            #############
            # Right column
            #############
        
            # # GW1 no weights, rely on historic data entirely.  Unsure how summer transfers will work out
            # game_week1_baseline_list = [[1,"Liverpool", "Bournemouth", 1, 1],
            #                   [2,"Aston Villa", "Newcastle", 1, 1],
            #                   [3,"Brighton", "Fulham", 1, 1],
            #                   [4,"Sunderland", "West Ham", 1, 1],
            #                   [5,"Spurs", "Burnley", 1, 1],
            #                   [6,"Wolves", "Man City", 1, 1],
            #                   [7,"Nott'm Forest", "Brentford", 1, 1], 
            #                   [8,"Chelsea", "Crystal Palace", 1, 1],
            #                   [9,"Man Utd", "Arsenal", 1, 1],
            #                   [10,"Leeds United", "Everton", 1, 1]]
            # game_week1_baseline = pd.DataFrame(game_week1_baseline_list, columns=["M#","Home", "Away", "H-Wt", "A-Wt"])
            
            # # GW2 Add weights now that we have seen everyone play once and better understand transfers   
            # game_week2_weighted_list = [[1,"West Ham", "Chelsea", 1, 1],
            #                   [2,"Man City", "Spurs", 1, 1],
            #                   [3,"Bournemouth", "Wolves", 1, 1],
            #                   [4,"Brentford", "Aston Villa", 1, 1],
            #                   [5,"Burnley", "Sunderland", 1, 1],
            #                   [6,"Arsenal", "Leeds United", 1, 1],
            #                   [7,"Crystal Palace", "Nott'm Forest", 1, 1], 
            #                   [8,"Everton", "Brighton", 2, 1],
            #                   [9,"Fulham", "Man Utd", 1, 1],
            #                   [10,"Newcastle", "Liverpool", 1, 1]]
            # game_week2_weighted = pd.DataFrame(game_week2_weighted_list, columns=["M#","Home", "Away", "H-Wt", "A-Wt"])
            
            # # GW3 Continue with weights plus observer underestimates in better teams from last season   
            # game_week3_weighted_list = [[1,"Chelsea", "Fulham", 1, 1],
            #                   [2, "Man Utd", "Burnley", 1, 1],
            #                   [3, "Sunderland", "Brentford", 1, 1],
            #                   [4, "Spurs", "Bournemouth",1, 1],
            #                   [5, "Wolves", "Everton",1 , 1],
            #                   [6, "Leeds United", "Newcastle",1, 1],
            #                   [7, "Brighton", "Man City", 1, 1], 
            #                   [8, "Nott'm Forest", "West Ham", 1, 2],
            #                   [9, "Liverpool", "Arsenal", 1, 1],
            #                   [10,"Aston Villa", "Crystal Palace", 1, 1]]
            # game_week3_weighted = pd.DataFrame(game_week3_weighted_list, columns=["M#","Home", "Away", "H-Wt", "A-Wt"])
            
            # # GW4 Continue with weights plus one more based on last 3 GW performance   
            # game_week4_weighted_list = [["-", 1,"Arsenal", "Nott'm Forest", 1, 1],
            #                   ["-", 2,"Bournemouth", "Brighton", 1, 1],
            #                   ["-", 3,"Crystal Palace", "Sunderland", 1, 1],
            #                   ["-", 4,"Everton", "Aston Villa",1, 1],
            #                   ["-", 5,"Fulham", "Leeds United",1 , 1],
            #                   ["-", 6,"Newcastle", "Wolves" ,1, 1],
            #                   ["-", 7,"West Ham", "Spurs", 1, 1], 
            #                   ["-", 8,"Brentford", "Chelsea", 1, 1],
            #                   ["-", 9,"Burnley", "Liverpool", 1, 1],
            #                   ["-", 10,"Man City", "Man Utd", 1, 1]]
            # game_week4_weighted = pd.DataFrame(game_week4_weighted_list, columns=["Date","M#","Home", "Away", "H-Wt", "A-Wt"])
            
            # # GW5 Continue with weights plus one more based on last 3 GW performance   
            # game_week5_list = [["Sat 20 Sep 04:30","Liverpool", "Everton"],
            #                   ["Sat 20 Sep 07:00", "Brighton", "Spurs"],
            #                   ["Sat 20 Sep 07:00", "Burnley", "Nott'm Forest"],
            #                   ["Sat 20 Sep 07:00", "West Ham", "Crystal Palace"],
            #                   ["Sat 20 Sep 07:00", "Wolves", "Leeds Utd"],
            #                   ["Sat 20 Sep 09:30", "Man Utd", "Chelsea"],
            #                   ["Sat 20 Sep 12:00", "Fulham", "Brentford"], 
            #                   ["Sun 21 Sep 06:00", "Bournemouth", "Newcastle"],
            #                   ["Sun 21 Sep 06:00", "Sunderland", "Aston Villa"],
            #                   ["Sun 21 Sep 08:30", "Arsenal", "Man City"]]
            # game_week5_df = pd.DataFrame(game_week5_list, columns=["Date","Home", "Away"])
            
            # # Create a dictionary to map string names to actual DataFrames
            # match_week_mapping = {
            #     "game_week1": game_week1_baseline,
            #     "game_week2": game_week2_weighted,
            #     "game_week3": game_week3_weighted,
            #     "game_week4": game_week4_weighted,
            #     "game_week5": game_week5_df
            # }
            
            # right.subheader("Game Week matches")
            
            # # pull down list to select game week to display
            # # gw_num_match_lineup = right.selectbox(
            # #     "Pick a game week:",
            # #     ("game_week1",
            # #      "game_week2",
            # #      "game_week3",
            # #      "game_week4",
            # #      "game_week5"
            # #      ),  key="gw_num_match_lineup")
            
            # # Get the selected DataFrame and display it
            # selected_dataframe = match_week_mapping.get(gw_num_pick)
            # if selected_dataframe is not None:
            #     if gw_num_pick in ('game_week5', 'game_week4'):
            #         right.dataframe(selected_dataframe, column_order=("Date","Home", "Away"), hide_index=True)
            #     else:
            #         right.dataframe(selected_dataframe, column_order=("Home", "Away"), hide_index=True)



    
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
    # Performance Dashboad
    #######################
    st.subheader("Performance Dashboard")
    create_football_performance_dashboard(timeseries_df)
    
    
    st.subheader("Team Performance Chart")
    
    import matplotlib.pyplot as plt
    
    # Group by team and sum W, D, L values (in case there are multiple rows per team)
    team_stats = merged_df.groupby('Team')[['W', 'D', 'L']].first().reset_index()
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
    
    # Display in streamlit
    st.pyplot(fig)

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
                create_heatmaps(timeseries_df)
            elif plot_name == "Simple":    
                simple_win_rate_heatmap(timeseries_df)
            elif plot_name == "Monthly Summary":
                monthly_summary_heatmap(timeseries_df)
    
                
    
    #st.image("team_performance.png")
    #st.bar_chart(team_stats, x="Team", y="Total_Games", color="Result", horizontal=True)
    
    st.subheader("Model Description")
    st.text("Model: Random Forest Classifer")
    
    
    st.subheader("Model Performance")
    
    #st.line_chart(accuracy_tracking, x="Game Week", y="Accuracy", x_label="Game Week", y_label="Accuracy %")
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
    
    
    
    
    
