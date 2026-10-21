#!/usr/bin/env python3
"""Unfreeze the top layers of a pretrained base model"""
from tensorflow import keras


def unfreeze_top_layers(model, n_layers):
    """
    Unfreezes the last n_layers of a base model and keeps the rest frozen.

    Args:
        model: keras Model that is the base model itself, or a full model
            that has the base model as its second layer.
        n_layers: number of layers at the end of the base model to unfreeze.

    Returns:
        None
    """
    base = model
    if len(model.layers) > 1 and isinstance(model.layers[1], keras.Model):
        base = model.layers[1]

    base.trainable = True
    n_total = len(base.layers)
    n_unfrozen = max(0, min(n_layers, n_total))

    for layer in base.layers[:n_total - n_unfrozen]:
        layer.trainable = False

    return None
