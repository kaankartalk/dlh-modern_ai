#!/usr/bin/env python3
"""Defines a function that builds an untrained Ridge regression model."""
from sklearn import linear_model


def ridge_regression(random_state):
    """Create an untrained Ridge Regression model.

    Ridge Regression extends ordinary linear regression by adding L2
    regularization, which helps stabilize the model by shrinking large
    coefficients.

    Args:
        random_state: An integer used to set the random seed for
            reproducibility.

    Returns:
        model: An untrained sklearn.linear_model.Ridge instance.
    """
    model = linear_model.Ridge(random_state=random_state)
    return model
