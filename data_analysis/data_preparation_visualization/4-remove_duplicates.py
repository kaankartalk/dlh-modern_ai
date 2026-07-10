#!/usr/bin/env python3
"""Remove duplicate rows from a DataFrame."""


def remove_duplicates(df):
    """Drop all duplicate rows.

    Args:
        df: pandas DataFrame to process.

    Returns:
        The deduplicated DataFrame.
    """
    df = df.drop_duplicates()

    return df
