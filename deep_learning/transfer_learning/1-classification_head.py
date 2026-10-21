#!/usr/bin/env python3
"""Classification head for a pretrained feature extractor"""
from tensorflow import keras


def add_classification_head(base_model, num_classes):
    """
    Attaches a classification head to a feature extractor.

    Args:
        base_model: keras Model whose output is a pooled feature vector.
        num_classes: number of output classes.

    Returns:
        A new keras Model ready for classification.
    """
    x = keras.layers.Dense(128, activation="relu")(base_model.output)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs=base_model.input, outputs=outputs)
