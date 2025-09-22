def highlight_multiple_conditions(row):
    """Apply different colors based on prediction being correct or not"""
    if row['Result'] == row['Predicted']:
        return ['background-color: #08f10a'] * len(row)  # Green
    elif row['Result'] != row['Predicted']:
        return ['background-color: #f13c08'] * len(row)  # Red
    else:
        return [''] * len(row)