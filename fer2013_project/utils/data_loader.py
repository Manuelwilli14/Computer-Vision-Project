"""
utils/data_loader.py
Load and preprocess the FER-2013 CSV dataset.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
NUM_CLASSES = 7
IMG_SIZE = 48


def load_fer2013(csv_path: str, use_original_split: bool = True):
    """
    Load FER-2013 CSV and return train/val/test splits as numpy arrays.

    Args:
        csv_path: Path to fer2013.csv
        use_original_split: If True, respect the original 'Usage' column splits.
                            If False, perform a stratified 70/15/15 split.

    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test), class_weight_dict
    """
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"Total samples : {len(df)}")
    print(f"Usage counts  :\n{df['Usage'].value_counts()}")
    print(f"Class counts  :\n{df['emotion'].value_counts().sort_index()}\n")

    # Parse pixel strings once
    df['pixels'] = df['pixels'].apply(
        lambda s: np.array(s.split(), dtype=np.float32)
    )

    if use_original_split:
        train_df = df[df['Usage'] == 'Training']
        val_df   = df[df['Usage'] == 'PublicTest']
        test_df  = df[df['Usage'] == 'PrivateTest']
    else:
        # Stratified custom split: 70 / 15 / 15
        train_df, tmp_df = train_test_split(
            df, test_size=0.30, random_state=42, stratify=df['emotion']
        )
        val_df, test_df = train_test_split(
            tmp_df, test_size=0.50, random_state=42, stratify=tmp_df['emotion']
        )

    X_train, y_train = _to_arrays(train_df)
    X_val,   y_val   = _to_arrays(val_df)
    X_test,  y_test  = _to_arrays(test_df)

    class_weight_dict = _compute_class_weights(train_df['emotion'].values)

    _print_summary(X_train, X_val, X_test, class_weight_dict)
    _sanity_check(X_train)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), class_weight_dict


def _to_arrays(subset: pd.DataFrame):
    """Reshape pixels → (N, 48, 48, 1) float32, labels → one-hot."""
    X = np.stack(subset['pixels'].values)
    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    y = to_categorical(subset['emotion'].values, num_classes=NUM_CLASSES)
    return X.astype(np.float32), y


def _compute_class_weights(emotion_labels: np.ndarray) -> dict:
    classes = np.arange(NUM_CLASSES)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=emotion_labels
    )
    return dict(enumerate(weights))


def _print_summary(X_train, X_val, X_test, cw):
    print("── Dataset splits ──────────────────────────────")
    print(f"  Train : {X_train.shape}")
    print(f"  Val   : {X_val.shape}")
    print(f"  Test  : {X_test.shape}")
    print("── Class weights ───────────────────────────────")
    for idx, label in enumerate(EMOTION_LABELS):
        print(f"  {idx} {label:<10} {cw[idx]:.3f}")
    print()


def _sanity_check(X_train):
    assert X_train.max() <= 1.0 and X_train.min() >= 0.0, "Normalisation failed"
    assert X_train.shape[1:] == (IMG_SIZE, IMG_SIZE, 1), "Wrong shape"
    print(f"Sanity check passed ✓  pixel range [{X_train.min():.3f}, {X_train.max():.3f}]\n")
