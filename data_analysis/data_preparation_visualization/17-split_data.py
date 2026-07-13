#!/usr/bin/env python3
"""Split data into stratified train/test sets."""
from sklearn import model_selection


def split_data(df, target='Churn', test_size=0.2, random_state=42):
    """Split a DataFrame into train/test sets using stratified sampling.

    Args:
        df: pandas DataFrame.
        target: Name of the target column.
        test_size: Proportion of the dataset for the test split.
        random_state: Random seed.

    Returns:
        A tuple (X_train, X_test, y_train, y_test).
    """
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)

    return X_train, X_test, y_train, y_test