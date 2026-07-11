#!/usr/bin/env python3
"""Visualize churn rate per category for a categorical column."""
import matplotlib.pyplot as plt


def plot_categorical_vs_churn(df, col):
    """Plot the churn rate per category for a categorical column.

    Args:
        df: pandas DataFrame with a 'Churn' column.
        col: Categorical column name.

    Returns:
        None
    """
    plt.figure(figsize=(12, 8))

    churn_rate = df.groupby(col)['Churn'].apply(lambda x: (x == 'Yes').mean())

    plt.bar(churn_rate.index, churn_rate.values)
    plt.title(f"Churn Rate by {col}")
    plt.ylabel("Churn Rate")
    plt.xticks(rotation=45)
    plt.show()

    return None
