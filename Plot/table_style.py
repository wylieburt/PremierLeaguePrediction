import pandas as pd

def highlight_multiple_conditions(row):
    """Apply different colors based on prediction being correct or not"""
    if pd.isna(row['Result']):
        return ['background-color: transparent'] * len(row) # No color
    if row['Result'] == row['Predicted']:
        return ['background-color: #08f10a'] * len(row)  # Green
    elif row['Result'] != row['Predicted']:
        return ['background-color: #f13c08'] * len(row)  # Red
    else:
        return [''] * len(row)
    
def highlight_max(row):
    # Create a Series of empty strings with the same index as the row
    styles = [''] * len(row)
    # Find the index of the maximum value
    max_idx = row.argmax()
    # Set the style for the max value
    styles[max_idx] = 'background-color: lightgreen'
    return styles