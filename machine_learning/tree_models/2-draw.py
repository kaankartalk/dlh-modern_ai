#!/usr/bin/env python3
"""View the decision rules of a trained tree."""
from sklearn.tree import export_text


def draw(clf, feature_names, class_names):
    """Print the textual structure of a trained decision tree."""
    tree_rules = export_text(
        clf,
        feature_names=list(feature_names),
        class_names=list(class_names)
    )
    print(tree_rules)