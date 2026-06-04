"""
model.py
CNN architecture for FER-2013 emotion classification.
"""

from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, MaxPooling2D,
    Dropout, Flatten, Dense,
)
from tensorflow.keras.regularizers import l2


IMG_SIZE   = 48
NUM_CLASSES = 7


def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=NUM_CLASSES) -> Sequential:
    """
    3-block convolutional network with BatchNorm and Dropout.

    Architecture per block:
        Conv2D → BatchNorm → Conv2D → BatchNorm → MaxPool → Dropout

    Head:
        Flatten → Dense(1024, ReLU) → Dropout(0.5) → Dense(7, softmax)
    """
    reg = l2(1e-4)

    model = Sequential([
        Input(shape=input_shape),

        # ── Block 1 — 32 filters ──────────────────────────────────
        Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=reg),
        BatchNormalization(),
        Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=reg),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # ── Block 2 — 64 filters ──────────────────────────────────
        Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=reg),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=reg),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # ── Block 3 — 128 filters ─────────────────────────────────
        Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=reg),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=reg),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # ── Dense head ────────────────────────────────────────────
        Flatten(),
        Dense(1024, activation='relu', kernel_regularizer=reg),
        BatchNormalization(),
        Dropout(0.5),

        Dense(num_classes, activation='softmax'),
    ], name='fer_cnn')

    return model


if __name__ == '__main__':
    m = build_model()
    m.summary()
