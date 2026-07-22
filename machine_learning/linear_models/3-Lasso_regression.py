#!/usr/bin/env python3
"""Defines a function that builds an untrained Lasso regression model."""
from sklearn import linear_model


def lasso_regression(random_state):
    """Create an untrained Lasso Regression model.

    Lasso Regression extends ordinary linear regression by adding L1
    regularization, which helps simplify the model by forcing some
    coefficients to zero, enabling automatic feature selection.

    Args:
        random_state: An integer used to set the random seed for
            reproducibility.

    Returns:
        model: An untrained sklearn.linear_model.Lasso instance.
    """
    model = linear_model.Lasso(random_state=random_state)
    return model
