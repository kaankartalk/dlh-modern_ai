#!/usr/bin/env python3
"""Module that visualizes missing values in a DataFrame."""
import matplotlib.pyplot as plt
import numpy as np


def plot_missingness(df):
    """Generate a scatter plot showing the location of missing values
    in a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to analyze.

    Returns:
        None
    """
    plt.figure(figsize=(12, 8))

    row_idx, col_idx = np.where(df.isnull().values)
    plt.scatter(row_idx, col_idx, marker='|')
    plt.yticks(range(len(df.columns)), df.columns)
    plt.title('Missingness Plot')

    plt.tight_layout()
    plt.show()
