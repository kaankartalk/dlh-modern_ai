#!/usr/bin/env python3
"""Retrieve the cost-complexity pruning path of a decision tree."""


def get_pruning_path(clf, X, y):
    """Return the ccp_alphas and impurities of the pruning path."""
    path = clf.cost_complexity_pruning_path(X, y)
    return path.ccp_alphas, path.impurities
