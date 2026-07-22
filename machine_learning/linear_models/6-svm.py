#!/usr/bin/env python3
"""Defines a function that builds an untrained SVM classifier."""
from sklearn import svm


def get_SVM_model(name, random_state):
    """Create an untrained Support Vector Machine (SVM) classifier.

    Args:
        name: A string indicating the kernel to use. Accepted values
            are 'linear', 'poly', and 'rbf'.
        random_state: The seed used by the random number generator for
            reproducibility.

    Returns:
        model: An untrained sklearn.svm.SVC instance using the
            specified kernel.
    """
    model = svm.SVC(kernel=name, random_state=random_state)
    return model
