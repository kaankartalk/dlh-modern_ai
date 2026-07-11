#!/usr/bin/env python3
"""Chi-square tests for categorical features against Churn."""
import pandas as pd
from scipy import stats


def chi_square_tests(df):
    """Compute chi-square p-values for categorical features vs Churn.

    Args:
        df: pandas DataFrame with 'Churn' and categorical columns.

    Returns:
        A dict mapping feature name to its chi-square p-value.
    """
    results = {}
    categorical_cols = df.select_dtypes(include='object').columns

    for col in categorical_cols:
        if col == 'Churn':
            continue
        contingency = pd.crosstab(df[col], df['Churn'])
        _, p_value, _, _ = stats.chi2_contingency(contingency)
        results[col] = p_value

    return results
