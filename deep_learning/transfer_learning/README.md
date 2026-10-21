# Transfer Learning

Exercises on transfer learning with Keras applications.

## Tasks

| File | Description |
|------|-------------|
| `0-frozen_extractor.py` | `build_feature_extractor()` builds a frozen MobileNetV2 feature extractor that outputs 1280-d vectors |
| `0-main.py` | Test script for task 0 |
| `1-classification_head.py` | `add_classification_head(base_model, num_classes)` adds a Dense(128, relu) layer and a softmax output layer on top of a feature extractor |
| `1-main.py` | Trains the full model on CIFAR-10 for 3 epochs |
| `2-unfreeze_top.py` | `unfreeze_top_layers(model, n_layers)` unfreezes the last `n_layers` of a base model and keeps the earlier layers frozen |
| `2-main.py` | Trains the head, then fine-tunes the top 20 backbone layers on CIFAR-10 |
| `3-data_aug.py` | `build_data_augmentation()` returns a seeded Sequential model with flip, rotation, zoom and contrast augmentation |
| `3-main.py` | Trains a model with augmentation on CIFAR-10 for 5 epochs |

## Requirements

- Python 3.9+
- TensorFlow 2.x

## Usage

    ./0-main.py
    ./1-main.py
    ./2-main.py
    ./3-main.py
