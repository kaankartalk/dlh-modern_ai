#!/usr/bin/env python3
"""Plot the distribution of the Churn column."""
import matplotlib.pyplot as plt


def plot_churn_distribution(df):
    """Plot a bar chart of Churn value counts.

    Args:
        df: pandas DataFrame with a 'Churn' column.

    Returns:
        None
    """
    plt.figure(figsize=(12, 8))

    counts = df['Churn'].value_counts()
    colors = ['skyblue' if label == 'No' else 'salmon'
              for label in counts.index]

    plt.bar(counts.index, counts.values, color=colors)
    plt.title('Churn Distribution')
    plt.ylabel('Count')
    plt.show()

    return None
