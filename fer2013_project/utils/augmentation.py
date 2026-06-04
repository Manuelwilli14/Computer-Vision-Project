"""
utils/augmentation.py
ImageDataGenerator configuration for FER-2013 training.
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_train_datagen() -> ImageDataGenerator:
    """Augmented generator for training set."""
    return ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.10,
        height_shift_range=0.10,
        horizontal_flip=True,
        zoom_range=0.10,
        shear_range=0.05,
        fill_mode='nearest',
    )


def get_val_datagen() -> ImageDataGenerator:
    """No augmentation for validation/test — just wraps numpy arrays."""
    return ImageDataGenerator()
