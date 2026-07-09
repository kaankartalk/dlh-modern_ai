#!/usr/bin/env python3
"""
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_missingness(df):
    """
    """
    plt.figure(figsize=(12, 8))

    n_rows, n_cols = df.shape

    row_idx, col_idx = np.where(df.isnull().values)
    plt.scatter(row_idx, col_idx, marker='|')

    plt.yticks(range(n_cols), df.columns)
    plt.xlim(-1, n_rows)
    plt.ylim(-1, n_cols)

    plt.tight_layout()
    plt.show()
    