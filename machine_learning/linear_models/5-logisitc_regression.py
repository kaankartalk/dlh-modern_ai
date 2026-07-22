#!/usr/bin/env python3
"""Defines a function that builds an untrained logistic regression model."""
from sklearn import linear_model


def Logistic_Regression_Model(random_state):
    """Create an untrained Logistic Regression model.

    Logistic Regression performs binary classification by fitting a
    logistic function to the data.

    Args:
        random_state: An integer used to set the random seed for
            reproducibility.

    Returns:
        model: An untrained sklearn.linear_model.LogisticRegression
            instance.
    """
    model = linear_model.LogisticRegression(random_state=random_state)
    return model