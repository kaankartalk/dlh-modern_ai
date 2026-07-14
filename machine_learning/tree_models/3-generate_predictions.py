#!/usr/bin/env python3
"""Generate predictions from a trained tree-based classifier."""


def generate_predictions(clf, X):
    """Return predicted class labels for the given feature matrix."""
    return clf.predict(X)
