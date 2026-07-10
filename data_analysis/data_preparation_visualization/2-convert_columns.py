
2 convert columns · PY
#!/usr/bin/env python3
"""Type conversion for the Telco Customer Churn DataFrame."""
 
import pandas as pd
 
 
def convert_columns(df):
    """Convert TotalCharges to numeric and SeniorCitizen to categorical.
 
    Args:
        df: pandas DataFrame containing 'TotalCharges' and 'SeniorCitizen'.
 
    Returns:
        The modified DataFrame.
    """
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
 
    return df
 