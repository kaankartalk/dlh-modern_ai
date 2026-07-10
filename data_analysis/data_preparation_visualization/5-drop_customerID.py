#!/usr/bin/env python3
"""Drop the customerID column."""


def drop_customerID(df):
    """Drop the customerID column since it lacks predictive value.

    Args:
        df: pandas DataFrame containing a 'customerID' column.

    Returns:
        The modified DataFrame.
    """
    df = df.drop(columns=['customerID'])

    return df
