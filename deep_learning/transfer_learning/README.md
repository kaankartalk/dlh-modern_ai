# Transfer Learning

Exercises on transfer learning with Keras applications.

## Tasks

| File | Description |
|------|-------------|
| `0-frozen_extractor.py` | `build_feature_extractor()` builds a frozen MobileNetV2 feature extractor that outputs 1280-d vectors |
| `0-main.py` | Test script for task 0 |
| `1-classification_head.py` | `add_classification_head(base_model, num_classes)` adds a Dense(128, relu) layer and a softmax output layer on top of a feature extractor |
| `1-main.py` | Trains the full model on CIFAR-10 for 3 epochs |

## Requirements

- Python 3.9+
- TensorFlow 2.x

## Usage

    ./0-main.py
    ./1-main.py
