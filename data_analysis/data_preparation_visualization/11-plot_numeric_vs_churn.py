#!/usr/bin/env python3
"""Compare continuous numeric feature distributions by churn."""
import matplotlib.pyplot as plt


def plot_numeric_vs_churn(df, col):
    """Plot overlapping histograms of a numeric column split by Churn.

    Args:
        df: pandas DataFrame with a 'Churn' column.
        col: Numeric column name.

    Returns:
        None
    """
    plt.figure(figsize=(12, 8))

    no = df[df['Churn'] == 'No'][col]
    yes = df[df['Churn'] == 'Yes'][col]

    plt.hist([no, yes], bins=30, label=['No', 'Yes'])

    plt.title(f"{col} Distribution by Churn")
    plt.xlabel(col)
    plt.legend(title='Churn')
    plt.show()

    return None
