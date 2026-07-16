#!/usr/bin/env python3
"""Compute feature importances from a trained random forest."""
import numpy as np


def feature_importance(rf):
    """Return feature importances and their indices sorted ascending."""
    importances = rf.feature_importances_
    indices = np.argsort(importances)
    return importances, indices
