#!/usr/bin/env python3
"""Train a tree-based classifier."""


def train_tree(clf, X, y):
    clf.fit(X, y)