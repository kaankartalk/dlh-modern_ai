#!/usr/bin/env python3
"""Frozen feature extractor built on MobileNetV2"""
from tensorflow import keras


def build_feature_extractor():
    """
    Builds a frozen MobileNetV2 feature extractor.

    Returns:
        A keras Model that maps (224, 224, 3) images to 1280-d feature vectors.
    """
    base_model = keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    outputs = keras.layers.GlobalAveragePooling2D()(x)

    return keras.Model(inputs, outputs)
