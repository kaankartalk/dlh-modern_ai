#!/usr/bin/env python3
"""Unfreeze the top layers of a pretrained base model"""


def unfreeze_top_layers(model, n_layers):
    """
    Unfreezes the last n_layers of a base model and keeps the rest frozen.

    Args:
        model: keras Model that contains the base model as a layer, or the
            base model itself.
        n_layers: number of layers at the end of the base model to unfreeze.

    Returns:
        None
    """
    base = model
    for layer in model.layers:
        if hasattr(layer, "layers"):
            base = layer
            break

    base.trainable = True
    n_total = len(base.layers)
    n_unfrozen = max(0, min(n_layers, n_total))

    for layer in base.layers[:n_total - n_unfrozen]:
        layer.trainable = False
    for layer in base.layers[n_total - n_unfrozen:]:
        layer.trainable = True

    return None
