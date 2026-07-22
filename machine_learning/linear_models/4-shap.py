#!/usr/bin/env python3
"""Defines a function that generates SHAP explanations for a model."""
import shap


def get_shap_explainer_and_values(model, X_train, X_test):
    """Create a SHAP explainer and compute SHAP values for predictions.

    Args:
        model: A trained regression model.
        X_train: Input data used to initialize the explainer.
        X_test: Input data to explain.

    Returns:
        tuple: (explainer, shap_values)
            explainer: SHAP explainer object.
            shap_values: SHAP values for the predictions on X_test.
    """
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    return explainer, shap_values
