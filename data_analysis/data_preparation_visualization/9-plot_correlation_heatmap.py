#!/usr/bin/env python3
"""Visualize correlations between continuous numeric features."""
import seaborn as sns
import matplotlib.pyplot as plt


def plot_correlation_heatmap(df):
    """Plot an annotated correlation heatmap for numeric features.

    Args:
        df: pandas DataFrame.

    Returns:
        None
    """
    plt.figure(figsize=(6, 5))

    corr = df.select_dtypes(include='number').corr()
    ticks = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                cbar_kws={'ticks': ticks})
    plt.title('Correlation Matrix')

    plt.tight_layout()
    plt.show()

    return None
