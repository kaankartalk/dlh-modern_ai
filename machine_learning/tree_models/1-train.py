#!/usr/bin/env python3
"""Train a tree-based classifier."""


def train_tree(clf, X, y):
    """Train a tree-based classifier on the given features and labels."""
    clf.fit(X, y)
