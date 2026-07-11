#!/usr/bin/env python3
"""Standardize numeric columns."""
from sklearn import preprocessing


def scale_numeric(df):
    """Scale MonthlyCharges and TotalCharges using StandardScaler.

    Args:
        df: pandas DataFrame.

    Returns:
        The modified DataFrame.
    """
    df = df.copy()

    cols = ['MonthlyCharges', 'TotalCharges']
    scaler = preprocessing.StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])

    return df
