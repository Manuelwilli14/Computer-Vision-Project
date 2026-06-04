"""
webcam_demo.py
Real-time facial emotion recognition using a webcam feed.

Usage:
    python webcam_demo.py
    python webcam_demo.py --model models/best_model.h5 --camera 0

Controls:
    Q  — quit
    S  — save current frame to outputs/snapshot.jpg
"""

import argparse
import os
import time
import cv2
import numpy as np
from tensorflow.keras.models import load_model

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
# BGR colour per emotion for the overlay rectangle
EMOTION_COLORS = {
    'Angry':    (0,   0,   220),
    'Disgust':  (0,  128,   0),
    'Fear':     (128,  0, 128),
    'Happy':    (0,  200,  200),
    'Neutral':  (180, 180, 180),
    'Sad':      (200, 100,   0),
    'Surprise': (0,  165,  255),
}

IMG_SIZE = 48


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model',  default='models/best_model.h5')
    p.add_argument('--camera', type=int, default=0)
    return p.parse_args()


def preprocess_face(gray_roi):
    """Resize ROI to 48×48 and normalise to [0,1]."""
    face = cv2.resize(gray_roi, (IMG_SIZE, IMG_SIZE))
    face = face.astype(np.float32) / 255.0
    return face.reshape(1, IMG_SIZE, IMG_SIZE, 1)


def draw_overlay(frame, x, y, w, h, label, confidence, color):
    """Draw bounding box, label and confidence bar."""
    # Rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Label background
    text     = f"{label}  {confidence:.0%}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(frame, (x, y - th - 10), (x + tw + 6, y), color, -1)

    # Label text (white)
    cv2.putText(frame, text, (x + 3, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # Confidence bar
    bar_w = int(w * confidence)
    cv2.rectangle(frame, (x, y + h + 4), (x + bar_w, y + h + 12), color, -1)
    cv2.rectangle(frame, (x, y + h + 4), (x + w,     y + h + 12), color, 1)


def main():
    args = parse_args()
    os.makedirs('outputs', exist_ok=True)

    # Load model
    print(f"Loading model: {args.model}")
    model = load_model(args.model)
    print("Model loaded ✓")

    # Haar cascade face detector
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")

    print("Press  Q  to quit,  S  to save a snapshot.")

    fps_timer  = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30),
        )

        for (x, y, w, h) in faces:
            roi = gray[y:y + h, x:x + w]
            inp = preprocess_face(roi)

            probs  = model.predict(inp, verbose=0)[0]
            idx    = np.argmax(probs)
            label  = EMOTION_LABELS[idx]
            conf   = float(probs[idx])
            color  = EMOTION_COLORS[label]

            draw_overlay(frame, x, y, w, h, label, conf, color)

        # FPS counter
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_timer = time.time()
        else:
            fps = frame_count / max(elapsed, 1e-6)

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow('Facial Emotion Recognition  —  Q to quit', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            path = 'outputs/snapshot.jpg'
            cv2.imwrite(path, frame)
            print(f"Snapshot saved → {path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
