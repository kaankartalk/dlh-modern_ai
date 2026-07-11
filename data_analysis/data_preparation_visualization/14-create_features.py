#!/usr/bin/env python3
"""Feature engineering: NumServices and TenureGroup."""
import pandas as pd


def create_features(df):
    """Create NumServices and TenureGroup features.

    Args:
        df: pandas DataFrame.

    Returns:
        The modified DataFrame with the new features, and the original
        columns used to build them dropped.
    """
    df = df.copy()

    service_cols = [
        'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies',
    ]

    num_services = pd.Series(0, index=df.index)
    for col in service_cols:
        if col == 'InternetService':
            num_services += df[col].isin(['DSL', 'Fiber optic']).astype(int)
        else:
            num_services += (df[col] == 'Yes').astype(int)

    df['NumServices'] = num_services

    bins = [0, 12, 24, 48, 60, float('inf')]
    labels = ['0-12', '13-24', '25-48', '49-60', '60+']
    df['TenureGroup'] = pd.cut(df['tenure'], bins=bins, labels=labels)

    df = df.drop(columns=service_cols + ['tenure'])

    return df
