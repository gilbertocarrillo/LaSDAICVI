"""The three Google-style keyword-spotting architectures of the thesis: CNN, GRU and CRNN.

Layer names and sizes follow the thesis models, so weights trained there load into these
builders (parameter counts for 12 classes: 2,769,420 / 593,836 / 467,772).
"""
from __future__ import annotations

import keras
from keras import layers

INPUT_SHAPE = (61, 20, 1)  # (MFCC frames, coefficients, channel)


def _head(x, num_classes: int):
    """Shared classifier: linear 128 -> ReLU 256 -> softmax."""
    x = layers.Dense(128, activation="linear", name="dense_linear")(x)
    x = layers.Dense(256, activation="relu", name="dense")(x)
    return layers.Dense(num_classes, activation="softmax", name="output")(x)


def _to_sequence(x):
    """Fold the feature and channel axes together so a GRU can read the frames as a sequence."""
    shape = x.shape  # [batch, time, feature, channels] -> [batch, time, feature * channels]
    return layers.Reshape((-1, shape[2] * shape[3]), name="reshape")(x)


def create_cnn_model(input_shape, num_classes) -> keras.Model:
    """Seven dilated convolutions, flatten, dropout 0.5, shared head (2.77 M parameters)."""
    filters = [64, 64, 64, 64, 128, 64, 128]
    kernels = [(3, 3), (5, 3), (5, 3), (5, 3), (5, 2), (5, 1), (10, 1)]
    dilations = [(1, 1), (1, 1), (2, 1), (1, 1), (2, 1), (1, 1), (2, 1)]
    inputs = keras.Input(shape=input_shape, name="input")
    x = inputs
    for i, (f, k, d) in enumerate(zip(filters, kernels, dilations), start=1):
        x = layers.Conv2D(f, k, activation="relu", dilation_rate=d, strides=(1, 1),
                          name=f"conv_{i}")(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dropout(0.5, name="dropout")(x)
    return keras.Model(inputs, _head(x, num_classes), name="cnn_google")


def create_gru_model(input_shape, num_classes) -> keras.Model:
    """One 400-unit GRU over the MFCC frames, dropout 0.1, shared head."""
    inputs = keras.Input(shape=input_shape, name="input")
    x = _to_sequence(inputs)
    x = layers.GRU(400, return_sequences=False, stateful=False, name="gru")(x)
    x = layers.Dropout(0.1, name="dropout")(x)
    return keras.Model(inputs, _head(x, num_classes), name="gru_google")


def create_crnn_model(input_shape, num_classes) -> keras.Model:
    """Two small convolutions followed by a 256-unit GRU, dropout 0.1, shared head."""
    inputs = keras.Input(shape=input_shape, name="input")
    x = inputs
    for i, (f, k) in enumerate(zip([16, 16], [(3, 3), (5, 3)]), start=1):
        x = layers.Conv2D(f, k, activation="relu", name=f"conv_{i}")(x)
    x = _to_sequence(x)
    x = layers.GRU(256, return_sequences=False, stateful=False, name="gru")(x)
    x = layers.Dropout(0.1, name="dropout")(x)
    return keras.Model(inputs, _head(x, num_classes), name="crnn_google")


ARCHS = {"cnn": create_cnn_model, "gru": create_gru_model, "crnn": create_crnn_model}


def build_model(arch: str, n_classes: int, input_shape=INPUT_SHAPE) -> keras.Model:
    """Instantiate `arch` ("cnn", "gru" or "crnn"); raises KeyError for anything else."""
    return ARCHS[arch](input_shape, n_classes)
