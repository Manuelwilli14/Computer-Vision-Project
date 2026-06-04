"""
train.py
Full training pipeline for FER-2013 emotion recognition CNN.

Usage:
    python train.py
    python train.py --csv data/fer2013.csv --epochs 100 --batch 64
"""

import argparse
import os
import numpy as np

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger,
)

from model import build_model
from utils import (
    load_fer2013,
    get_train_datagen,
    get_val_datagen,
    plot_training_history,
)


def parse_args():
    p = argparse.ArgumentParser(description='Train FER-2013 CNN')
    p.add_argument('--csv',    default='data/fer2013.csv', help='Path to fer2013.csv')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch',  type=int, default=64)
    p.add_argument('--lr',     type=float, default=1e-3)
    p.add_argument('--no-augment', action='store_true', help='Disable data augmentation')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs('models',  exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    # ── 1. Load data ─────────────────────────────────────────────
    (X_train, y_train), (X_val, y_val), (X_test, y_test), class_weight_dict = \
        load_fer2013(args.csv)

    # ── 2. Build model ───────────────────────────────────────────
    model = build_model()
    model.compile(
        optimizer=Adam(learning_rate=args.lr),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    model.summary()

    # ── 3. Data generators ───────────────────────────────────────
    if args.no_augment:
        train_gen = get_val_datagen().flow(X_train, y_train, batch_size=args.batch, shuffle=True)
    else:
        train_gen = get_train_datagen().flow(X_train, y_train, batch_size=args.batch, shuffle=True)

    val_gen = get_val_datagen().flow(X_val, y_val, batch_size=args.batch, shuffle=False)

    # ── 4. Callbacks ─────────────────────────────────────────────
    callbacks = [
        ModelCheckpoint(
            'models/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        CSVLogger('outputs/training_log.csv'),
    ]

    # ── 5. Train ─────────────────────────────────────────────────
    steps_per_epoch  = len(X_train) // args.batch
    validation_steps = len(X_val)   // args.batch

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        epochs=args.epochs,
        class_weight=class_weight_dict,
        callbacks=callbacks,
    )

    # ── 6. Final evaluation ──────────────────────────────────────
    print("\n── Test set evaluation ──────────────────────────────────")
    loss, acc = model.evaluate(X_test, y_test, batch_size=args.batch, verbose=0)
    print(f"  Test loss     : {loss:.4f}")
    print(f"  Test accuracy : {acc:.4f} ({acc*100:.2f}%)")

    # ── 7. Save curves ───────────────────────────────────────────
    plot_training_history(history, save_path='outputs/training_curves.png')
    print("\nTraining complete. Best model saved to models/best_model.h5")


if __name__ == '__main__':
    main()
