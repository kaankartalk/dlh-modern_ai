#!/usr/bin/env python3
"""Handle missing values in TotalCharges."""

import pandas as pd


def clean_total_charges(df, method='drop'):
    """Handle missing values in the TotalCharges column.

    Args:
        df: pandas DataFrame with missing values in 'TotalCharges'.
        method: Strategy to use, one of 'drop', 'median', 'impute'.

    Returns:
        The modified DataFrame.
    """
    if method == 'drop':
        df = df.dropna(subset=['TotalCharges'])
    elif method == 'median':
        median = df['TotalCharges'].median()
        df['TotalCharges'] = df['TotalCharges'].fillna(median)
    elif method == 'impute':
        imputed = df['MonthlyCharges'] * df['tenure']
        df['TotalCharges'] = df['TotalCharges'].fillna(imputed)

    return df
