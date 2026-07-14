#!/usr/bin/env python3
"""Evaluate classifier performance."""
from sklearn import metrics


def evaluate(true_labels, predicted_labels, class_names):
    """Return a classification report comparing true and predicted labels."""
    return metrics.classification_report(
        true_labels,
        predicted_labels,
        target_names=list(class_names)
    )
