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
