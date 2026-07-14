#!/usr/bin/env python3
"""Select the best ccp_alpha for post-pruning a decision tree."""


def get_best_alpha(clfs, train_scores, test_scores, ccp_alphas):
    """Return the ccp_alpha and classifier with the best generalization."""
    best_test_score = max(test_scores)

    candidates = [
        i for i in range(len(clfs))
        if test_scores[i] == best_test_score
    ]

    smallest_gap = min(
        abs(train_scores[i] - test_scores[i]) for i in candidates
    )
    candidates = [
        i for i in candidates
        if abs(train_scores[i] - test_scores[i]) == smallest_gap
    ]

    best_index = max(candidates, key=lambda i: ccp_alphas[i])

    return ccp_alphas[best_index], clfs[best_index]
