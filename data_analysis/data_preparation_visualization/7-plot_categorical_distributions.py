#!/usr/bin/env python3
"""Visualize categorical feature distributions."""
import matplotlib.pyplot as plt


def plot_categorical_distributions(df, columns_to_plot=None):
    """Plot bar charts for categorical features in a grid layout.

    Args:
        df: pandas DataFrame.
        columns_to_plot: Optional list of categorical columns.
            Defaults to all object columns, excluding 'Churn'.

    Returns:
        None
    """
    if columns_to_plot is None:
        columns_to_plot = [
            col for col in df.select_dtypes(include='object').columns
            if col != 'Churn'
        ]
    else:
        columns_to_plot = columns_to_plot

    n_cols, n_rows = 3, -(-len(columns_to_plot) // 3)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for ax, col in zip(axes, columns_to_plot):
        counts = df[col].value_counts()
        ax.bar(counts.index, counts.values)
        ax.set_title(col)
        ax.tick_params(axis='x', labelrotation=45)

    for ax in axes[len(columns_to_plot):]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("Task_7.png")
    plt.show()

    return None
