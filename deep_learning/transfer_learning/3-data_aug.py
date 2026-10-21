#!/usr/bin/env python3
"""Data augmentation pipeline for transfer learning"""
from tensorflow import keras


def build_data_augmentation():
    """
    Builds a Sequential model with common image augmentation layers.

    All layers are seeded with 42 for reproducibility.

    Returns:
        A keras Sequential model that augments images during training.
    """
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal", seed=42),
        keras.layers.RandomRotation(0.15, seed=42),
        keras.layers.RandomZoom(0.15, seed=42),
        keras.layers.RandomContrast(0.1, seed=42),
    ])
