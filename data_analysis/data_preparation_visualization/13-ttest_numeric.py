#!/usr/bin/env python3
"""Welch's t-tests for numeric features against Churn."""
from scipy import stats


def ttest_numeric(df):
    """Compute Welch's t-test p-values for numeric features vs Churn.

    Args:
        df: pandas DataFrame with a 'Churn' column.

    Returns:
        A dict mapping feature name to its t-test p-value.
    """
    results = {}
    numeric_cols = df.select_dtypes(include='number').columns

    for col in numeric_cols:
        yes = df[df['Churn'] == 'Yes'][col]
        no = df[df['Churn'] == 'No'][col]
        _, p_value = stats.ttest_ind(yes, no, equal_var=False)
        results[col] = p_value

    return results
