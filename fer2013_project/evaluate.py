"""
evaluate.py
Load a trained model and evaluate it on the FER-2013 test set.

Usage:
    python evaluate.py
    python evaluate.py --model models/best_model.h5 --csv data/fer2013.csv
"""

import argparse
import numpy as np
from tensorflow.keras.models import load_model

from utils import load_fer2013, plot_confusion_matrix, plot_sample_predictions


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='models/best_model.h5')
    p.add_argument('--csv',   default='data/fer2013.csv')
    p.add_argument('--batch', type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()

    # Load data (only test split needed here)
    _, _, (X_test, y_test), _ = load_fer2013(args.csv)

    # Load model
    print(f"Loading model: {args.model}")
    model = load_model(args.model)

    # Predict
    y_prob = model.predict(X_test, batch_size=args.batch, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Accuracy
    acc = (y_pred == y_true).mean()
    print(f"\nTest accuracy: {acc:.4f} ({acc*100:.2f}%)")

    # Plots
    plot_confusion_matrix(y_true, y_pred,
                          save_path='outputs/confusion_matrix.png',
                          normalize=True)

    plot_sample_predictions(X_test, y_true, y_pred, n=20,
                            save_path='outputs/sample_predictions.png')


if __name__ == '__main__':
    main()
