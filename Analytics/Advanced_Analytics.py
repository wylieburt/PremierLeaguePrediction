import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')


def test(name):
    return f"Hello, {name}.  I am available for us of functions"

def list_functions():
    function_list = ["advanced_football_analytics_suite(df)",
                     "create_streak_analysis(daily_results)",
                     "create_momentum_analysis(daily_results)",
                     "create_predictive_models(daily_results)",
                     "create_statistical_distributions(daily_results)",
                     "create_cyclical_patterns(df, daily_results)",
                     "create_performance_correlations(daily_results)",
                     "create_volatility_analysis(daily_results)",
                     "create_scenario_analysis(daily_results)"]
                     
    return function_list

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')

def advanced_football_analytics_suite(df):
    """
    Comprehensive advanced analytics for English football timeseries data
    """
    print("🏈 ADVANCED FOOTBALL ANALYTICS SUITE")
    print("="*50)
    
    # Prepare data
    daily_results = df.groupby(['Date', 'Result'])['count'].sum().unstack(fill_value=0)
    for col in ['W', 'L', 'D']:
        if col not in daily_results.columns:
            daily_results[col] = 0
    
    daily_results['Total'] = daily_results[['W', 'L', 'D']].sum(axis=1)
    daily_results['Win_Rate'] = daily_results['W'] / daily_results['Total']
    daily_results['Points'] = daily_results['W'] * 3 + daily_results['D']
    daily_results['PPG'] = daily_results['Points'] / daily_results['Total']
    
    # Run all analytics
    streak_fig = create_streak_analysis(daily_results)
    momentum_fig = create_momentum_analysis(daily_results)
    predictive_fig = create_predictive_models(daily_results)
    stats_fig = create_statistical_distributions(daily_results)
    cyclical_fig = create_cyclical_patterns(df, daily_results)
    performance_fig = create_performance_correlations(daily_results)
    volatility_fig = create_volatility_analysis(daily_results)
    scenario_fig = create_scenario_analysis(daily_results)
    
    return streak_fig, momentum_fig, predictive_fig, stats_fig, cyclical_fig, performance_fig, volatility_fig, scenario_fig

def create_streak_analysis(daily_results):
    """Analyze winning/losing streaks and momentum"""
    print("\n1. 📈 STREAK & MOMENTUM ANALYSIS")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Streak & Momentum Analysis', fontsize=16, fontweight='bold')
    
    # Calculate streaks
    streaks = calculate_streaks(daily_results)
    
    # 1. Streak Timeline
    ax1 = axes[0, 0]
    colors = ['red' if x < 0 else 'green' if x > 0 else 'orange' for x in streaks['streak']]
    bars = ax1.bar(range(len(streaks)), streaks['streak'], color=colors, alpha=0.7)
    ax1.set_title('Winning/Losing Streaks Over Time')
    ax1.set_xlabel('Match Day')
    ax1.set_ylabel('Streak Length (+ Win, - Loss)')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.grid(True, alpha=0.3)
    
    # 2. Streak Distribution
    ax2 = axes[0, 1]
    win_streaks = [x for x in streaks['streak'] if x > 0]
    loss_streaks = [abs(x) for x in streaks['streak'] if x < 0]
    
    if win_streaks:
        ax2.hist(win_streaks, bins=range(1, max(win_streaks)+2), alpha=0.7, 
                label=f'Win Streaks (avg: {np.mean(win_streaks):.1f})', color='green')
    if loss_streaks:
        ax2.hist(loss_streaks, bins=range(1, max(loss_streaks)+2), alpha=0.7,
                label=f'Loss Streaks (avg: {np.mean(loss_streaks):.1f})', color='red')
    
    ax2.set_title('Streak Length Distribution')
    ax2.set_xlabel('Streak Length')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Momentum Indicator
    ax3 = axes[1, 0]
    momentum = calculate_momentum(daily_results)
    ax3.plot(daily_results.index, momentum, linewidth=2, color='purple')
    ax3.fill_between(daily_results.index, momentum, 0, alpha=0.3, color='purple')
    ax3.set_title('Momentum Indicator (Weighted Recent Form)')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Momentum Score')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.3)
    
    # 4. Streak Recovery Analysis
    ax4 = axes[1, 1]
    recovery_analysis = analyze_streak_recovery(streaks)
    if recovery_analysis and any(v > 0 for v in recovery_analysis.values()):
        categories = list(recovery_analysis.keys())
        values = list(recovery_analysis.values())
        
        # Filter out zero values
        non_zero_data = [(cat, val) for cat, val in zip(categories, values) if val > 0]
        if non_zero_data:
            categories, values = zip(*non_zero_data)
            colors_recovery = ['darkred', 'red', 'orange', 'lightgreen', 'green'][:len(categories)]
            ax4.pie(values, labels=categories, autopct='%1.1f%%', colors=colors_recovery)
            ax4.set_title('Recovery Patterns After Poor Form')
        else:
            ax4.text(0.5, 0.5, 'Insufficient losing streaks\nfor recovery analysis', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Recovery Patterns After Poor Form')
    else:
        ax4.text(0.5, 0.5, 'Insufficient data\nfor recovery analysis', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Recovery Patterns After Poor Form')
    
    plt.tight_layout()
    #plt.show()
    
    return fig
    # Print streak insights
    #print_streak_insights(streaks, daily_results)

def create_momentum_analysis(daily_results):
    """Advanced momentum and trend analysis"""
    print("\n2. 🚀 MOMENTUM & TREND ANALYSIS")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Momentum & Trend Analysis', fontsize=16, fontweight='bold')
    
    # 1. Multiple Moving Averages
    ax1 = axes[0, 0]
    ma_short = daily_results['PPG'].rolling(window=5).mean()
    ma_medium = daily_results['PPG'].rolling(window=15).mean()
    ma_long = daily_results['PPG'].rolling(window=30).mean()
    
    ax1.plot(daily_results.index, ma_short, label='5-game MA', linewidth=2, alpha=0.8)
    ax1.plot(daily_results.index, ma_medium, label='15-game MA', linewidth=2, alpha=0.8)
    ax1.plot(daily_results.index, ma_long, label='30-game MA', linewidth=2, alpha=0.8)
    ax1.fill_between(daily_results.index, ma_short, ma_long, alpha=0.2)
    
    ax1.set_title('Moving Average Convergence/Divergence')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Points Per Game')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Trend Strength Analysis
    ax2 = axes[0, 1]
    trend_strength = calculate_trend_strength(daily_results['PPG'])
    if len(trend_strength) > 0:
        colors_trend = ['red' if x < -0.3 else 'orange' if x < 0 else 'lightgreen' if x < 0.3 else 'green' 
                       for x in trend_strength]
        ax2.scatter(range(len(trend_strength)), trend_strength, c=colors_trend, alpha=0.7)
        ax2.set_title('Trend Strength Over Time')
        ax2.set_xlabel('Period')
        ax2.set_ylabel('Trend Strength (-1 to 1)')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Insufficient data\nfor trend analysis', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Trend Strength Over Time')
    
    # 3. Performance Acceleration
    ax3 = axes[1, 0]
    acceleration = calculate_acceleration(daily_results['PPG'])
    if len(acceleration) > 0:
        # Create date index for acceleration (which is 2 elements shorter)
        accel_dates = daily_results.index[2:2+len(acceleration)]
        ax3.plot(accel_dates, acceleration, linewidth=2, color='blue')
        ax3.fill_between(accel_dates, acceleration, 0, alpha=0.3, color='blue')
        ax3.set_title('Performance Acceleration (Rate of Change)')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Acceleration')
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Insufficient data\nfor acceleration analysis', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Performance Acceleration (Rate of Change)')
    
    # 4. Momentum Oscillator
    ax4 = axes[1, 1]
    oscillator = calculate_momentum_oscillator(daily_results)
    ax4.plot(daily_results.index, oscillator, linewidth=2, color='purple')
    ax4.fill_between(daily_results.index, oscillator, 50, 
                    where=(oscillator > 50), alpha=0.3, color='green', label='Bullish')
    ax4.fill_between(daily_results.index, oscillator, 50, 
                    where=(oscillator <= 50), alpha=0.3, color='red', label='Bearish')
    ax4.set_title('Momentum Oscillator (0-100)')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Oscillator Value')
    ax4.axhline(y=50, color='black', linestyle='--', alpha=0.5)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # plt.show()
    return fig

def create_predictive_models(daily_results):
    """Build predictive models for future performance"""
    print("\n3. 🔮 PREDICTIVE MODELING")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Predictive Models & Forecasting', fontsize=16, fontweight='bold')
    
    # Prepare features for modeling
    features_df = create_features_for_modeling(daily_results)
    
    # 1. Linear Trend Prediction
    ax1 = axes[0, 0]
    X = np.arange(len(daily_results)).reshape(-1, 1)
    y = daily_results['PPG'].values
    model = LinearRegression().fit(X, y)
    
    # Predict next 20 periods
    future_X = np.arange(len(daily_results), len(daily_results) + 20).reshape(-1, 1)
    future_pred = model.predict(future_X)
    
    ax1.plot(daily_results.index, y, label='Actual PPG', alpha=0.7)
    ax1.plot(daily_results.index, model.predict(X), label='Trend Line', linestyle='--', color='red')
    
    # Plot future predictions
    future_dates = pd.date_range(start=daily_results.index[-1], periods=21)[1:]
    ax1.plot(future_dates, future_pred, label='Forecast', linestyle=':', color='green', linewidth=2)
    
    ax1.set_title(f'Linear Trend Forecast (R²: {model.score(X, y):.3f})')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Points Per Game')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Win Probability Model
    ax2 = axes[0, 1]
    win_prob = calculate_win_probability(daily_results)
    ax2.plot(daily_results.index, win_prob * 100, linewidth=2, color='orange')
    ax2.fill_between(daily_results.index, win_prob * 100, 50, 
                    where=(win_prob > 0.5), alpha=0.3, color='green')
    ax2.fill_between(daily_results.index, win_prob * 100, 50, 
                    where=(win_prob <= 0.5), alpha=0.3, color='red')
    ax2.set_title('Dynamic Win Probability')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Win Probability (%)')
    ax2.axhline(y=50, color='black', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # 3. Confidence Intervals
    ax3 = axes[1, 0]
    rolling_mean = daily_results['PPG'].rolling(window=10).mean()
    rolling_std = daily_results['PPG'].rolling(window=10).std()
    upper_bound = rolling_mean + 1.96 * rolling_std
    lower_bound = rolling_mean - 1.96 * rolling_std
    
    ax3.plot(daily_results.index, rolling_mean, label='Mean PPG', linewidth=2)
    ax3.fill_between(daily_results.index, upper_bound, lower_bound, alpha=0.2, label='95% Confidence')
    ax3.scatter(daily_results.index, daily_results['PPG'], alpha=0.5, s=10)
    
    ax3.set_title('Performance Confidence Intervals')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Points Per Game')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Regime Detection
    ax4 = axes[1, 1]
    regimes = detect_performance_regimes(daily_results['PPG'])
    colors_regime = plt.cm.Set3(np.linspace(0, 1, len(set(regimes))))
    
    for i, regime in enumerate(set(regimes)):
        mask = np.array(regimes) == regime
        ax4.scatter(np.arange(len(daily_results))[mask], daily_results['PPG'].values[mask], 
                   c=[colors_regime[i]], label=f'Regime {regime}', alpha=0.7)
    
    ax4.set_title('Performance Regime Detection')
    ax4.set_xlabel('Time Period')
    ax4.set_ylabel('Points Per Game')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # plt.show()
    return fig

def create_statistical_distributions(daily_results):
    """Analyze statistical distributions and patterns"""
    print("\n4. 📊 STATISTICAL DISTRIBUTION ANALYSIS")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Statistical Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 1. PPG Distribution
    ax1 = axes[0, 0]
    ppg_values = daily_results['PPG'].dropna()
    if len(ppg_values) > 0:
        ax1.hist(ppg_values, bins=20, alpha=0.7, density=True, color='skyblue')
        
        # Fit normal distribution
        try:
            mu, sigma = stats.norm.fit(ppg_values)
            x = np.linspace(ppg_values.min(), ppg_values.max(), 100)
            ax1.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
                    label=f'Normal (μ={mu:.2f}, σ={sigma:.2f})')
            ax1.legend()
        except Exception as e:
            print(f"Normal distribution fit failed: {e}")
    
    ax1.set_title('Points Per Game Distribution')
    ax1.set_xlabel('PPG')
    ax1.set_ylabel('Density')
    ax1.grid(True, alpha=0.3)
    
    # 2. Win Rate Distribution
    ax2 = axes[0, 1]
    win_rates = daily_results['Win_Rate'].dropna()
    ax2.hist(win_rates, bins=20, alpha=0.7, density=True, color='lightgreen')
    
    # Fit beta distribution (need to handle edge cases)
    if len(win_rates) > 10 and win_rates.var() > 0:
        try:
            # Clip values slightly away from 0 and 1 for beta fitting
            win_rates_clipped = np.clip(win_rates, 0.01, 0.99)
            a, b, loc, scale = stats.beta.fit(win_rates_clipped, floc=0, fscale=1)
            x = np.linspace(0.01, 0.99, 100)
            ax2.plot(x, stats.beta.pdf(x, a, b), 'r-', linewidth=2, 
                    label=f'Beta (α={a:.2f}, β={b:.2f})')
        except Exception as e:
            # Fallback to normal distribution if beta fails
            try:
                mu, sigma = stats.norm.fit(win_rates)
                x = np.linspace(win_rates.min(), win_rates.max(), 100)
                ax2.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
                        label=f'Normal (μ={mu:.2f}, σ={sigma:.2f})')
            except:
                pass
    
    ax2.set_title('Win Rate Distribution')
    ax2.set_xlabel('Win Rate')
    ax2.set_ylabel('Density')
    if len(ax2.get_legend_handles_labels()[0]) > 0:
        ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Q-Q Plot for Normality
    ax3 = axes[0, 2]
    if len(ppg_values) > 3:
        try:
            stats.probplot(ppg_values, dist="norm", plot=ax3)
            ax3.set_title('Q-Q Plot: PPG vs Normal')
            ax3.grid(True, alpha=0.3)
        except:
            ax3.text(0.5, 0.5, 'Q-Q plot failed\nInsufficient valid data', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Q-Q Plot: PPG vs Normal')
    else:
        ax3.text(0.5, 0.5, 'Insufficient data\nfor Q-Q analysis', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Q-Q Plot: PPG vs Normal')
    
    # 4. Autocorrelation
    ax4 = axes[1, 0]
    if len(ppg_values) > 15:
        try:
            autocorr = calculate_autocorrelation(daily_results['PPG'])
            lags = range(len(autocorr))
            ax4.bar(lags, autocorr, alpha=0.7, color='orange')
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax4.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='Significance')
            ax4.axhline(y=-0.2, color='red', linestyle='--', alpha=0.5)
            ax4.set_title('Autocorrelation Function')
            ax4.set_xlabel('Lag')
            ax4.set_ylabel('Autocorrelation')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        except:
            ax4.text(0.5, 0.5, 'Autocorrelation\nanalysis failed', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Autocorrelation Function')
    else:
        ax4.text(0.5, 0.5, 'Insufficient data\nfor autocorrelation', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Autocorrelation Function')
    
    # 5. Rolling Variance
    ax5 = axes[1, 1]
    rolling_var = daily_results['PPG'].rolling(window=10).var()
    ax5.plot(daily_results.index, rolling_var, linewidth=2, color='purple')
    ax5.fill_between(daily_results.index, rolling_var, alpha=0.3, color='purple')
    ax5.set_title('Rolling Variance (Consistency)')
    ax5.set_xlabel('Date')
    ax5.set_ylabel('Variance')
    ax5.grid(True, alpha=0.3)
    
    # 6. Outlier Detection
    ax6 = axes[1, 2]
    if len(ppg_values) > 3:
        try:
            z_scores = np.abs(stats.zscore(ppg_values))
            outliers = z_scores > 2
            
            ax6.scatter(range(len(ppg_values)), ppg_values, alpha=0.6, color='blue', label='Normal')
            if np.any(outliers):
                ax6.scatter(np.arange(len(ppg_values))[outliers], ppg_values[outliers], 
                           color='red', s=50, label=f'Outliers (|z|>2): {np.sum(outliers)}')
            ax6.set_title('Outlier Detection')
            ax6.set_xlabel('Match Day')
            ax6.set_ylabel('PPG')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        except:
            ax6.text(0.5, 0.5, 'Outlier detection\nfailed', 
                    ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Outlier Detection')
    else:
        ax6.text(0.5, 0.5, 'Insufficient data\nfor outlier detection', 
                ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Outlier Detection')
    
    plt.tight_layout()
    # plt.show()
    
    return fig
    # Print statistical summary
    #print_statistical_summary(daily_results)

def create_cyclical_patterns(df, daily_results):
    """Analyze cyclical and seasonal patterns"""
    print("\n5. 🔄 CYCLICAL & SEASONAL PATTERNS")
    
    # Add time features
    df_analysis = df.copy()
    df_analysis['Date'] = pd.to_datetime(df_analysis['Date'])
    df_analysis['Month'] = df_analysis['Date'].dt.month
    df_analysis['DayOfWeek'] = df_analysis['Date'].dt.dayofweek
    df_analysis['Season'] = df_analysis['Date'].apply(lambda x: get_football_season(x))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Cyclical & Seasonal Pattern Analysis', fontsize=16, fontweight='bold')
    
    # 1. Monthly Performance Heatmap
    ax1 = axes[0, 0]
    monthly_perf = analyze_monthly_patterns(df_analysis)
    if not monthly_perf.empty:
        sns.heatmap(monthly_perf, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax1, center=1.5)
    ax1.set_title('Monthly Performance Heatmap (PPG)')
    
    # 2. Day of Week Analysis
    ax2 = axes[0, 1]
    dow_analysis = analyze_day_of_week_patterns(df_analysis)
    if not dow_analysis.empty:
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        # Create arrays for all 7 days, filling missing days with 0
        ppg_values = []
        day_labels = []
        
        for day_idx in range(7):
            if day_idx in dow_analysis.index:
                ppg_values.append(dow_analysis.loc[day_idx, 'PPG'])
                day_labels.append(days[day_idx])
            else:
                ppg_values.append(0)
                day_labels.append(days[day_idx])
        
        bars = ax2.bar(range(7), ppg_values, color='lightblue', alpha=0.7)
        ax2.set_xticks(range(7))
        ax2.set_xticklabels(day_labels)
        ax2.set_title('Performance by Day of Week')
        ax2.set_ylabel('Average PPG')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars for non-zero values
        for i, (bar, value) in enumerate(zip(bars, ppg_values)):
            if value > 0:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{value:.2f}', ha='center', va='bottom', fontsize=9)
    else:
        ax2.text(0.5, 0.5, 'No day-of-week\ndata available', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Performance by Day of Week')
    
    # 3. Season Start vs End Performance
    ax3 = axes[0, 2]
    season_phases = analyze_season_phases(df_analysis)
    if season_phases and any(season_phases[p]['Total'] > 0 for p in season_phases):
        phases = list(season_phases.keys())
        values = [season_phases[p]['PPG'] for p in phases]
        colors = ['red', 'orange', 'green'][:len(phases)]
        
        bars = ax3.bar(phases, values, color=colors, alpha=0.7)
        ax3.set_title('Performance by Season Phase')
        ax3.set_ylabel('Average PPG')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=9)
    else:
        ax3.text(0.5, 0.5, 'Insufficient data\nfor season phases', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Performance by Season Phase')
    
    # 4. Fourier Transform (Frequency Analysis)
    ax4 = axes[1, 0]
    if len(daily_results) > 20:
        try:
            frequencies, power = analyze_frequency_components(daily_results['PPG'])
            if len(frequencies) > 0 and len(power) > 0:
                ax4.plot(frequencies, power, linewidth=2)
                ax4.set_title('Frequency Analysis (Cyclical Patterns)')
                ax4.set_xlabel('Frequency')
                ax4.set_ylabel('Power')
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'No significant\nfrequency patterns', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Frequency Analysis (Cyclical Patterns)')
        except:
            ax4.text(0.5, 0.5, 'Frequency analysis\nfailed', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Frequency Analysis (Cyclical Patterns)')
    else:
        ax4.text(0.5, 0.5, 'Insufficient data\nfor frequency analysis', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Frequency Analysis (Cyclical Patterns)')
    
    # 5. Home/Away Effect (if applicable)
    ax5 = axes[1, 1]
    # Simulate home/away effect for demonstration
    home_away_effect = simulate_home_away_analysis(df_analysis)
    if home_away_effect:
        categories = list(home_away_effect.keys())
        values = list(home_away_effect.values())
        ax5.bar(categories, values, color=['green', 'red'], alpha=0.7)
        ax5.set_title('Home vs Away Performance')
        ax5.set_ylabel('Average PPG')
        ax5.grid(True, alpha=0.3)
    
    # 6. Calendar Effect
    ax6 = axes[1, 2]
    calendar_effect = analyze_calendar_effects(df_analysis)
    if calendar_effect and len(calendar_effect) > 2:
        months = list(calendar_effect.keys())
        win_rates = [calendar_effect[m]['Win_Rate'] for m in months]
        
        ax6.plot(months, win_rates, marker='o', linewidth=2, markersize=6)
        ax6.set_title('Calendar Effect on Win Rate')
        ax6.set_xlabel('Month')
        ax6.set_ylabel('Win Rate (%)')
        ax6.grid(True, alpha=0.3)
        
        # Set month labels if we have them
        month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax6.set_xticks(months)
        ax6.set_xticklabels([month_names[m] for m in months])
    else:
        ax6.text(0.5, 0.5, 'Insufficient data\nfor calendar analysis', 
                ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Calendar Effect on Win Rate')
    
    plt.tight_layout()
    # plt.show()
    return fig

def create_performance_correlations(daily_results):
    """Analyze correlations between different performance metrics"""
    print("\n6. 🔗 PERFORMANCE CORRELATIONS")
    
    # Create correlation matrix with robust handling
    correlation_data = daily_results[['Win_Rate', 'PPG', 'Total']].copy()
    
    # Add additional metrics with error handling
    try:
        correlation_data['Volatility'] = daily_results['PPG'].rolling(window=max(5, len(daily_results)//10)).std()
    except:
        correlation_data['Volatility'] = 0
    
    try:
        correlation_data['Momentum'] = calculate_momentum(daily_results)
    except:
        correlation_data['Momentum'] = 0
        
    try:
        window_size = max(5, len(daily_results)//5)
        correlation_data['Trend'] = daily_results['PPG'].rolling(window=window_size).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window_size and not np.any(np.isnan(x)) else 0,
            raw=False
        )
    except:
        correlation_data['Trend'] = 0
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Performance Correlation Analysis', fontsize=16, fontweight='bold')
    
    # 1. Correlation Heatmap
    ax1 = axes[0, 0]
    try:
        corr_matrix = correlation_data.corr()
        # Filter out NaN values
        corr_matrix = corr_matrix.fillna(0)
        
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax1, 
                    square=True, cbar_kws={'shrink': 0.8}, fmt='.2f')
        ax1.set_title('Performance Metrics Correlation')
    except Exception as e:
        ax1.text(0.5, 0.5, 'Correlation matrix\nfailed to generate', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Performance Metrics Correlation')
    
    # 2. Scatter Plot Matrix
    ax2 = axes[0, 1]
    try:
        valid_data = daily_results[['Total', 'PPG']].dropna()
        if len(valid_data) > 3:
            ax2.scatter(valid_data['Total'], valid_data['PPG'], alpha=0.6, color='blue')
            
            # Add trend line if we have enough data
            if len(valid_data) > 10:
                z = np.polyfit(valid_data['Total'], valid_data['PPG'], 1)
                p = np.poly1d(z)
                ax2.plot(valid_data['Total'], p(valid_data['Total']), "r--", alpha=0.8)
            
            ax2.set_xlabel('Games Played')
            ax2.set_ylabel('Points Per Game')
            ax2.set_title('Games vs Performance Correlation')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Insufficient data\nfor scatter plot', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Games vs Performance Correlation')
    except:
        ax2.text(0.5, 0.5, 'Scatter plot\nfailed', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Games vs Performance Correlation')
    
    # 3. Lag Correlation Analysis
    ax3 = axes[1, 0]
    if len(daily_results) > 15:
        try:
            lag_correlations = []
            for lag in range(1, min(11, len(daily_results) // 2)):
                ppg_series = daily_results['PPG'].dropna()
                if len(ppg_series) > lag:
                    corr = ppg_series.corr(ppg_series.shift(lag))
                    lag_correlations.append(corr if not pd.isna(corr) else 0)
                else:
                    lag_correlations.append(0)
            
            if len(lag_correlations) > 0:
                ax3.bar(range(1, len(lag_correlations) + 1), lag_correlations, alpha=0.7, color='orange')
                ax3.set_title('Lag Correlation (Performance Persistence)')
                ax3.set_xlabel('Lag (Days)')
                ax3.set_ylabel('Correlation')
                ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax3.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='Significance')
                ax3.axhline(y=-0.2, color='red', linestyle='--', alpha=0.5)
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'Lag correlation\nanalysis failed', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Lag Correlation (Performance Persistence)')
        except:
            ax3.text(0.5, 0.5, 'Lag correlation\nanalysis failed', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Lag Correlation (Performance Persistence)')
    else:
        ax3.text(0.5, 0.5, 'Insufficient data\nfor lag analysis', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Lag Correlation (Performance Persistence)')
    
    # 4. Rolling Correlation
    ax4 = axes[1, 1]
    if len(daily_results) > 40:  # Need sufficient data for rolling correlation
        try:
            rolling_corr = daily_results['Win_Rate'].rolling(window=20).corr(daily_results['PPG'])
            rolling_corr = rolling_corr.dropna()
            
            if len(rolling_corr) > 0:
                ax4.plot(rolling_corr.index, rolling_corr, linewidth=2, color='purple')
                ax4.set_title('Rolling Correlation: Win Rate vs PPG')
                ax4.set_xlabel('Date')
                ax4.set_ylabel('Correlation')
                ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax4.set_ylim(-1, 1)
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'Rolling correlation\ncalculation failed', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Rolling Correlation: Win Rate vs PPG')
        except Exception as e:
            # Fallback: calculate simple correlation over time periods
            try:
                period_size = max(10, len(daily_results) // 5)
                periods = []
                correlations = []
                
                for i in range(period_size, len(daily_results), period_size):
                    period_data = daily_results.iloc[i-period_size:i]
                    if len(period_data) > 3:
                        corr = period_data['Win_Rate'].corr(period_data['PPG'])
                        if not pd.isna(corr):
                            periods.append(period_data.index[-1])
                            correlations.append(corr)
                
                if len(correlations) > 0:
                    ax4.plot(periods, correlations, marker='o', linewidth=2, color='purple')
                    ax4.set_title('Period Correlation: Win Rate vs PPG')
                    ax4.set_xlabel('Date')
                    ax4.set_ylabel('Correlation')
                    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                    ax4.set_ylim(-1, 1)
                    ax4.grid(True, alpha=0.3)
                else:
                    ax4.text(0.5, 0.5, 'Correlation analysis\nfailed', 
                            ha='center', va='center', transform=ax4.transAxes)
                    ax4.set_title('Rolling Correlation: Win Rate vs PPG')
            except:
                ax4.text(0.5, 0.5, 'Correlation analysis\nfailed', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Rolling Correlation: Win Rate vs PPG')
    else:
        ax4.text(0.5, 0.5, 'Insufficient data\nfor rolling correlation', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Rolling Correlation: Win Rate vs PPG')
    
    plt.tight_layout()
    # plt.show()
    return fig

def create_volatility_analysis(daily_results):
    """Analyze performance volatility and risk metrics"""
    print("\n7. 📈 VOLATILITY & RISK ANALYSIS")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Volatility & Risk Analysis', fontsize=16, fontweight='bold')
    
    # 1. Rolling Volatility
    ax1 = axes[0, 0]
    volatility = daily_results['PPG'].rolling(window=10).std()
    ax1.plot(daily_results.index, volatility, linewidth=2, color='red')
    ax1.fill_between(daily_results.index, volatility, alpha=0.3, color='red')
    ax1.set_title('Rolling Volatility (10-game window)')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Standard Deviation')
    ax1.grid(True, alpha=0.3)
    
    # 2. Value at Risk (VaR)
    ax2 = axes[0, 1]
    var_95 = daily_results['PPG'].rolling(window=20).quantile(0.05)
    var_99 = daily_results['PPG'].rolling(window=20).quantile(0.01)
    
    ax2.plot(daily_results.index, daily_results['PPG'], alpha=0.5, label='PPG', color='blue')
    ax2.plot(daily_results.index, var_95, color='orange', linewidth=2, label='VaR 95%')
    ax2.plot(daily_results.index, var_99, color='red', linewidth=2, label='VaR 99%')
    ax2.fill_between(daily_results.index, var_99, var_95, alpha=0.2, color='red')
    
    ax2.set_title('Value at Risk Analysis')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Points Per Game')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Drawdown Analysis
    ax3 = axes[1, 0]
    cumulative_ppg = daily_results['PPG'].cumsum()
    running_max = cumulative_ppg.expanding().max()
    drawdown = (cumulative_ppg - running_max) / running_max * 100
    
    ax3.fill_between(daily_results.index, drawdown, 0, alpha=0.7, color='red')
    ax3.plot(daily_results.index, drawdown, linewidth=1, color='darkred')
    ax3.set_title('Performance Drawdown Analysis')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Drawdown (%)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Risk-Return Scatter
    ax4 = axes[1, 1]
    rolling_return = daily_results['PPG'].rolling(window=10).mean()
    rolling_risk = daily_results['PPG'].rolling(window=10).std()
    
    scatter = ax4.scatter(rolling_risk, rolling_return, alpha=0.6, c=range(len(rolling_risk)), 
                         cmap='viridis')
    ax4.set_xlabel('Risk (Standard Deviation)')
    ax4.set_ylabel('Return (Average PPG)')
    ax4.set_title('Risk-Return Profile Over Time')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='Time Period')
    
    plt.tight_layout()
    # plt.show()
    
    return fig

def create_scenario_analysis(daily_results):
    """Perform scenario analysis and what-if simulations"""
    print("\n8. 🎯 SCENARIO ANALYSIS & SIMULATIONS")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Scenario Analysis & What-If Simulations', fontsize=16, fontweight='bold')
    
    # 1. Monte Carlo Simulation
    ax1 = axes[0, 0]
    monte_carlo_results = run_monte_carlo_simulation(daily_results, n_simulations=1000, periods=30)
    
    # Plot percentiles
    percentiles = [5, 25, 50, 75, 95]
    colors = ['red', 'orange', 'blue', 'lightgreen', 'green']
    for i, p in enumerate(percentiles):
        ax1.plot(range(30), np.percentile(monte_carlo_results, p, axis=0), 
                color=colors[i], label=f'{p}th percentile', linewidth=2)
    
    ax1.fill_between(range(30), np.percentile(monte_carlo_results, 5, axis=0),
                    np.percentile(monte_carlo_results, 95, axis=0), alpha=0.2, color='gray')
    
    ax1.set_title('Monte Carlo Simulation (30 games ahead)')
    ax1.set_xlabel('Games into Future')
    ax1.set_ylabel('Cumulative Points')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Best/Worst Case Scenarios
    ax2 = axes[0, 1]
    scenarios = calculate_scenarios(daily_results)
    
    scenario_names = list(scenarios.keys())
    final_positions = [scenarios[s]['final_points'] for s in scenario_names]
    colors_scenario = ['red', 'orange', 'green']
    
    bars = ax2.bar(scenario_names, final_positions, color=colors_scenario, alpha=0.7)
    ax2.set_title('Season Ending Scenarios')
    ax2.set_ylabel('Projected Final Points')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, final_positions):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Performance Target Analysis
    ax3 = axes[1, 0]
    targets = [1.0, 1.5, 2.0, 2.5]  # PPG targets
    probabilities = []
    
    for target in targets:
        prob = calculate_target_probability(daily_results, target)
        probabilities.append(prob * 100)
    
    colors_target = ['red', 'orange', 'lightgreen', 'green']
    bars_target = ax3.bar([f'{t} PPG' for t in targets], probabilities, 
                         color=colors_target, alpha=0.7)
    
    ax3.set_title('Probability of Achieving Targets')
    ax3.set_ylabel('Probability (%)')
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.3)
    
    # Add percentage labels
    for bar, prob in zip(bars_target, probabilities):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{prob:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 4. Sensitivity Analysis
    ax4 = axes[1, 1]
    sensitivity_results = perform_sensitivity_analysis(daily_results)
    
    factors = list(sensitivity_results.keys())
    impacts = list(sensitivity_results.values())
    colors_sens = ['red' if x < 0 else 'green' for x in impacts]
    
    bars_sens = ax4.barh(factors, impacts, color=colors_sens, alpha=0.7)
    ax4.set_title('Sensitivity Analysis (Impact on Final Points)')
    ax4.set_xlabel('Points Impact')
    ax4.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # plt.show()
    
    return fig
    # Print scenario insights
    #print_scenario_insights(scenarios, daily_results)

# Helper functions for all the advanced analytics

def calculate_streaks(daily_results):
    """Calculate winning and losing streaks"""
    streaks = []
    current_streak = 0
    current_type = None
    
    for idx, row in daily_results.iterrows():
        if row['W'] > row['L'] and row['W'] > row['D']:
            result = 'W'
        elif row['L'] > row['W'] and row['L'] > row['D']:
            result = 'L'
        else:
            result = 'D'
        
        if result == current_type:
            if result == 'W':
                current_streak += 1
            elif result == 'L':
                current_streak -= 1
            # Draws don't extend streaks
        else:
            current_type = result
            if result == 'W':
                current_streak = 1
            elif result == 'L':
                current_streak = -1
            else:
                current_streak = 0
        
        streaks.append({'date': idx, 'streak': current_streak, 'result': result})
    
    return pd.DataFrame(streaks)

def calculate_momentum(daily_results):
    """Calculate momentum indicator using exponential weighting"""
    if len(daily_results) < 4:
        return [0] * len(daily_results)
    
    weights = np.array([0.4, 0.3, 0.2, 0.1])  # More recent games weighted higher
    momentum = []
    
    for i in range(len(daily_results)):
        if i < 3:
            momentum.append(0)
        else:
            recent_ppg = daily_results['PPG'].iloc[i-3:i+1].values
            if len(recent_ppg) == 4 and not np.any(np.isnan(recent_ppg)):
                weighted_avg = np.average(recent_ppg, weights=weights)
                overall_avg = daily_results['PPG'].iloc[:i+1].mean()
                if not np.isnan(overall_avg) and overall_avg != 0:
                    momentum_score = (weighted_avg - overall_avg) * 10  # Scale for visibility
                    momentum.append(momentum_score)
                else:
                    momentum.append(0)
            else:
                momentum.append(0)
    
    return momentum

def analyze_streak_recovery(streaks):
    """Analyze recovery patterns after losing streaks"""
    recovery_patterns = {
        'Quick Recovery (1-2 games)': 0,
        'Moderate Recovery (3-5 games)': 0,
        'Slow Recovery (6-10 games)': 0,
        'Extended Struggle (>10 games)': 0,
        'No Clear Pattern': 0
    }
    
    if len(streaks) == 0:
        return recovery_patterns
    
    losing_streaks = []
    for i, row in streaks.iterrows():
        if row['streak'] <= -3:  # Losing streak of 3 or more
            losing_streaks.append(i)
    
    if len(losing_streaks) == 0:
        return recovery_patterns
    
    for streak_start in losing_streaks:
        recovery_time = 0
        for j in range(streak_start + 1, min(streak_start + 15, len(streaks))):
            if j < len(streaks) and streaks.iloc[j]['streak'] > 0:
                recovery_time = j - streak_start
                break
        
        if recovery_time == 0:
            recovery_patterns['No Clear Pattern'] += 1
        elif recovery_time <= 2:
            recovery_patterns['Quick Recovery (1-2 games)'] += 1
        elif recovery_time <= 5:
            recovery_patterns['Moderate Recovery (3-5 games)'] += 1
        elif recovery_time <= 10:
            recovery_patterns['Slow Recovery (6-10 games)'] += 1
        else:
            recovery_patterns['Extended Struggle (>10 games)'] += 1
    
    return recovery_patterns

def calculate_trend_strength(series, window=10):
    """Calculate trend strength using linear regression slope"""
    if len(series) < window:
        return [0] * (len(series) - window + 1) if len(series) >= window else []
    
    trend_strengths = []
    
    for i in range(window, len(series) + 1):
        y = series.iloc[i-window:i].values
        x = np.arange(window)
        
        if len(y) == window and not np.any(np.isnan(y)) and np.var(y) > 0:
            try:
                slope, _, r_value, _, _ = stats.linregress(x, y)
                trend_strength = slope * r_value  # Slope weighted by correlation
                trend_strengths.append(trend_strength)
            except:
                trend_strengths.append(0)
        else:
            trend_strengths.append(0)
    
    return trend_strengths

def calculate_acceleration(series):
    """Calculate acceleration (second derivative) of performance"""
    series_clean = series.dropna()
    if len(series_clean) < 3:
        return np.array([])
    
    velocity = np.diff(series_clean)
    acceleration = np.diff(velocity)
    return acceleration

def calculate_momentum_oscillator(daily_results, short_window=5, long_window=15):
    """Calculate momentum oscillator (0-100 scale)"""
    short_ma = daily_results['PPG'].rolling(window=short_window).mean()
    long_ma = daily_results['PPG'].rolling(window=long_window).mean()
    
    oscillator = ((short_ma - long_ma) / long_ma) * 100 + 50
    return oscillator.fillna(50)

def create_features_for_modeling(daily_results):
    """Create features for predictive modeling"""
    features = daily_results.copy()
    
    # Lag features
    for lag in [1, 2, 3, 5]:
        features[f'PPG_lag_{lag}'] = daily_results['PPG'].shift(lag)
        features[f'Win_Rate_lag_{lag}'] = daily_results['Win_Rate'].shift(lag)
    
    # Rolling statistics
    for window in [5, 10, 15]:
        features[f'PPG_rolling_mean_{window}'] = daily_results['PPG'].rolling(window).mean()
        features[f'PPG_rolling_std_{window}'] = daily_results['PPG'].rolling(window).std()
    
    # Trend features
    features['PPG_trend'] = daily_results['PPG'].rolling(window=10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0
    )
    
    return features

def calculate_win_probability(daily_results):
    """Calculate dynamic win probability based on recent form"""
    rolling_win_rate = daily_results['Win_Rate'].rolling(window=10, min_periods=1).mean()
    rolling_ppg = daily_results['PPG'].rolling(window=10, min_periods=1).mean()
    
    # Combine win rate and PPG for probability (normalized)
    prob = (rolling_win_rate * 0.6) + ((rolling_ppg / 3.0) * 0.4)
    return prob.fillna(0.5)

def detect_performance_regimes(series, n_regimes=3):
    """Detect different performance regimes using quantiles"""
    quantiles = np.quantile(series.dropna(), np.linspace(0, 1, n_regimes + 1))
    regimes = []
    
    for value in series:
        if pd.isna(value):
            regimes.append(0)
        else:
            regime = 0
            for i in range(len(quantiles) - 1):
                if quantiles[i] <= value <= quantiles[i + 1]:
                    regime = i
                    break
            regimes.append(regime)
    
    return regimes

def calculate_autocorrelation(series, max_lags=15):
    """Calculate autocorrelation function"""
    series_clean = series.dropna()
    autocorr = []
    
    for lag in range(max_lags):
        if lag == 0:
            autocorr.append(1.0)
        else:
            corr = series_clean.corr(series_clean.shift(lag))
            autocorr.append(corr if not pd.isna(corr) else 0)
    
    return autocorr

def get_football_season(date):
    """Convert date to football season"""
    if date.month >= 8:
        return f"{date.year}-{str(date.year + 1)[2:]}"
    else:
        return f"{date.year - 1}-{str(date.year)[2:]}"

def analyze_monthly_patterns(df):
    """Analyze performance patterns by month"""
    monthly_stats = df.groupby(['Month', 'Result'])['count'].sum().unstack(fill_value=0)
    
    if 'W' in monthly_stats.columns and 'L' in monthly_stats.columns and 'D' in monthly_stats.columns:
        monthly_stats['Total'] = monthly_stats[['W', 'L', 'D']].sum(axis=1)
        monthly_stats['PPG'] = (monthly_stats['W'] * 3 + monthly_stats['D']) / monthly_stats['Total']
        return monthly_stats[['PPG']].T
    
    return pd.DataFrame()

def analyze_day_of_week_patterns(df):
    """Analyze performance by day of week"""
    dow_stats = df.groupby(['DayOfWeek', 'Result'])['count'].sum().unstack(fill_value=0)
    
    if not dow_stats.empty:
        for col in ['W', 'L', 'D']:
            if col not in dow_stats.columns:
                dow_stats[col] = 0
        
        dow_stats['Total'] = dow_stats[['W', 'L', 'D']].sum(axis=1)
        dow_stats['PPG'] = (dow_stats['W'] * 3 + dow_stats['D']) / dow_stats['Total']
        return dow_stats
    
    return pd.DataFrame()

def analyze_season_phases(df):
    """Analyze performance by season phases"""
    df['Month'] = pd.to_datetime(df['Date']).dt.month
    
    phases = {
        'Early Season': [8, 9, 10],
        'Mid Season': [11, 12, 1, 2],
        'Late Season': [3, 4, 5]
    }
    
    phase_stats = {}
    for phase, months in phases.items():
        phase_data = df[df['Month'].isin(months)]
        if not phase_data.empty:
            phase_results = phase_data.groupby('Result')['count'].sum()
            total = phase_results.sum()
            if total > 0:
                wins = phase_results.get('W', 0)
                draws = phase_results.get('D', 0)
                ppg = (wins * 3 + draws) / total
                phase_stats[phase] = {'PPG': ppg, 'Total': total}
    
    return phase_stats

def analyze_frequency_components(series):
    """Analyze frequency components using FFT"""
    series_clean = series.dropna().values
    
    if len(series_clean) > 10:
        fft = np.fft.fft(series_clean)
        frequencies = np.fft.fftfreq(len(series_clean))
        power = np.abs(fft) ** 2
        
        # Keep only positive frequencies
        pos_mask = frequencies > 0
        return frequencies[pos_mask][:len(frequencies)//4], power[pos_mask][:len(power)//4]
    
    return np.array([]), np.array([])

def simulate_home_away_analysis(df):
    """Simulate home/away effect for demonstration"""
    # In real analysis, you'd have actual home/away data
    total_games = df['count'].sum()
    
    # Simulate that home games have slightly better performance
    home_performance = 1.8  # Simulated home PPG
    away_performance = 1.4  # Simulated away PPG
    
    return {'Home': home_performance, 'Away': away_performance}

def analyze_calendar_effects(df):
    """Analyze calendar effects on performance"""
    df['Month'] = pd.to_datetime(df['Date']).dt.month
    monthly_stats = df.groupby(['Month', 'Result'])['count'].sum().unstack(fill_value=0)
    
    calendar_effects = {}
    for month in range(1, 13):
        if month in monthly_stats.index:
            month_data = monthly_stats.loc[month]
            total = month_data.sum()
            if total > 0:
                wins = month_data.get('W', 0)
                win_rate = (wins / total) * 100
                calendar_effects[month] = {'Win_Rate': win_rate}
    
    return calendar_effects

def run_monte_carlo_simulation(daily_results, n_simulations=1000, periods=30):
    """Run Monte Carlo simulation for future performance"""
    ppg_mean = daily_results['PPG'].mean()
    ppg_std = daily_results['PPG'].std()
    
    simulations = []
    for _ in range(n_simulations):
        simulation = []
        cumulative_points = 0
        
        for period in range(periods):
            # Sample from normal distribution
            ppg = np.random.normal(ppg_mean, ppg_std)
            ppg = max(0, min(3, ppg))  # Bound between 0 and 3
            cumulative_points += ppg
            simulation.append(cumulative_points)
        
        simulations.append(simulation)
    
    return np.array(simulations)

def calculate_scenarios(daily_results):
    """Calculate best/worst/expected case scenarios"""
    current_ppg = daily_results['PPG'].mean()
    games_remaining = 38 - len(daily_results)  # Assuming 38-game season
    
    scenarios = {
        'Worst Case': {
            'ppg': current_ppg * 0.7,  # 30% drop in performance
            'final_points': daily_results['Points'].sum() + (games_remaining * current_ppg * 0.7)
        },
        'Expected': {
            'ppg': current_ppg,
            'final_points': daily_results['Points'].sum() + (games_remaining * current_ppg)
        },
        'Best Case': {
            'ppg': current_ppg * 1.3,  # 30% improvement
            'final_points': daily_results['Points'].sum() + (games_remaining * current_ppg * 1.3)
        }
    }
    
    return scenarios

def calculate_target_probability(daily_results, target_ppg):
    """Calculate probability of achieving target PPG"""
    current_mean = daily_results['PPG'].mean()
    current_std = daily_results['PPG'].std()
    
    # Calculate z-score
    z_score = (target_ppg - current_mean) / current_std if current_std > 0 else 0
    
    # Return probability (1 - CDF for targets above mean)
    if target_ppg <= current_mean:
        return 1 - stats.norm.cdf(z_score)
    else:
        return stats.norm.cdf(-z_score)

def perform_sensitivity_analysis(daily_results):
    """Perform sensitivity analysis on key factors"""
    baseline_points = daily_results['Points'].sum()
    
    # Simulate impact of various factors
    sensitivity = {
        'Win Rate +10%': baseline_points * 1.2 - baseline_points,
        'Win Rate -10%': baseline_points * 0.8 - baseline_points,
        'More Draws': baseline_points * 1.05 - baseline_points,
        'Fewer Draws': baseline_points * 0.95 - baseline_points,
        'Home Advantage': baseline_points * 1.15 - baseline_points
    }
    
    return sensitivity

def print_streak_insights(streaks, daily_results):
    """Print insights about streaks"""
    max_win_streak = streaks['streak'].max()
    max_loss_streak = abs(streaks['streak'].min())
    
    print(f"📊 STREAK ANALYSIS INSIGHTS:")
    print(f"   Longest winning streak: {max_win_streak} games")
    print(f"   Longest losing streak: {max_loss_streak} games")
    print(f"   Current form: Last 5 games show {daily_results['PPG'].tail(5).mean():.2f} PPG")

def print_statistical_summary(daily_results):
    """Print statistical summary"""
    print(f"📈 STATISTICAL SUMMARY:")
    print(f"   Mean PPG: {daily_results['PPG'].mean():.2f}")
    print(f"   PPG Volatility: {daily_results['PPG'].std():.2f}")
    print(f"   Skewness: {daily_results['PPG'].skew():.2f}")
    print(f"   Kurtosis: {daily_results['PPG'].kurtosis():.2f}")

def print_scenario_insights(scenarios, daily_results):
    """Print scenario analysis insights"""
    print(f"🎯 SCENARIO INSIGHTS:")
    for scenario, data in scenarios.items():
        print(f"   {scenario}: {data['final_points']:.0f} points ({data['ppg']:.2f} PPG)")

# Create sample data and run analysis
def create_sample_football_data():
    """Create sample football data for demonstration"""
    np.random.seed(42)
    dates = pd.date_range('2023-08-15', '2024-05-15', freq='D')
    sample_data = []
    
    for date in dates:
        if date.weekday() in [1, 5, 6]:  # Match days
            results = np.random.choice(['W', 'D', 'L'], 
                                     size=np.random.randint(1, 3),
                                     p=[0.45, 0.30, 0.25])
            
            for result in results:
                sample_data.append({
                    'Date': date.strftime('%Y-%m-%d'),
                    'Result': result,
                    'count': 1
                })
    
    return pd.DataFrame(sample_data)

# Main execution
if __name__ == "__main__":
    print("🏈 LAUNCHING ADVANCED FOOTBALL ANALYTICS...")
    print("="*60)
    
    # Create and preprocess sample data
    all_data_df = pd.read_csv("match_data_20_26.csv")
    all_data_df['Result'] = all_data_df['Result'].str.split(' ').str[0]

    timeseries_df = all_data_df.groupby(["Date", "Result"])["Result"].value_counts()
    
    football_df = timeseries_df.reset_index()
    
    
    football_df = create_sample_football_data()
    football_df['Date'] = pd.to_datetime(football_df['Date'])
    
    print(f"Sample dataset: {len(football_df)} match records")
    print(f"Date range: {football_df['Date'].min()} to {football_df['Date'].max()}")
    
    # Run comprehensive analytics
    advanced_football_analytics_suite(football_df)
    
    print("\n" + "="*60)
    print("🎉 ADVANCED ANALYTICS COMPLETE!")
    print("="*60)
    print("To use with your real data:")
    print("1. Load your data: df = pd.read_csv('your_data.csv')")
    print("2. Run: advanced_football_analytics_suite(df)")
    print("\nThis suite provides:")
    print("✅ Streak & momentum analysis")
    print("✅ Predictive modeling & forecasting") 
    print("✅ Statistical distribution analysis")
    print("✅ Cyclical & seasonal patterns")
    print("✅ Performance correlations")
    print("✅ Volatility & risk metrics")
    print("✅ Scenario analysis & simulations")
    print("✅ Monte Carlo forecasting")