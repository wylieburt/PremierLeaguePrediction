import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


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
    
    return fig

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
    
    return plt
    

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
    
    return plt


