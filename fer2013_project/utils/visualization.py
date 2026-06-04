"""
utils/visualization.py
Plotting helpers: training curves, confusion matrix, sample grid.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


def plot_training_history(history, save_path: str = None):
    """Plot accuracy and loss curves side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle('Training history', fontsize=14, fontweight='bold')

    # Accuracy
    axes[0].plot(history.history['accuracy'],   label='Train',      linewidth=1.8)
    axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=1.8)
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history['loss'],     label='Train',      linewidth=1.8)
    axes[1].plot(history.history['val_loss'], label='Validation', linewidth=1.8)
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves → {save_path}")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, save_path: str = None, normalize: bool = True):
    """
    Plot confusion matrix.

    Args:
        y_true: 1-D array of integer class indices (ground truth)
        y_pred: 1-D array of integer class indices (predictions)
        save_path: optional file path to save the figure
        normalize: if True, show percentages instead of counts
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt, title = '.2f', 'Confusion matrix (normalised)'
    else:
        cm_display = cm
        fmt, title = 'd', 'Confusion matrix (counts)'

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=EMOTION_LABELS,
        yticklabels=EMOTION_LABELS,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix → {save_path}")
    plt.show()

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=EMOTION_LABELS))


def plot_sample_predictions(X, y_true, y_pred, n: int = 20, save_path: str = None):
    """
    Display a grid of sample images with true vs predicted labels.
    Correct predictions shown in green, incorrect in red.
    """
    indices = np.random.choice(len(X), size=min(n, len(X)), replace=False)
    cols = 5
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.8))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        img = X[idx].squeeze()
        true_lbl = EMOTION_LABELS[y_true[idx]]
        pred_lbl = EMOTION_LABELS[y_pred[idx]]
        correct  = y_true[idx] == y_pred[idx]

        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        color = '#2e7d32' if correct else '#c62828'
        axes[i].set_title(f"T: {true_lbl}\nP: {pred_lbl}", fontsize=8, color=color)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle('Sample predictions  (green=correct, red=wrong)', fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved sample predictions → {save_path}")
    plt.show()


def plot_class_distribution(emotion_series, title: str = 'Class distribution', save_path: str = None):
    """Bar chart of emotion class frequencies."""
    counts = emotion_series.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(EMOTION_LABELS, counts.values, color='steelblue', edgecolor='white')
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_title(title, fontsize=13)
    ax.set_ylabel('Sample count')
    ax.set_xlabel('Emotion')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
