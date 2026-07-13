#!/usr/bin/env python3
"""Build a Decision Tree Classifier."""
from sklearn import tree


def build_decision_tree(min_samples_leaf, min_samples_split, random_state):
    """
    Creates a Decision Tree Classifier using Gini impurity, no max depth.

    Args:
        min_samples_leaf: minimum number of samples required at a leaf node.
        min_samples_split: minimum number of samples required to split
            an internal node.
        random_state: seed for the random number generator.

    Returns:
        model: a Scikit-learn DecisionTreeClassifier instance.
    """
    model = tree.DecisionTreeClassifier(
        criterion='gini',
        max_depth=None,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        random_state=random_state
    )
    return model
