"""
utils.py — Helper functions for YOLO detection pipeline
"""

import os
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np


# ── Source type detection ─────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v"}


def get_source_type(source: str) -> str:
    """
    Returns: 'image' | 'video' | 'folder' | 'webcam'
    """
    if source.isdigit():
        return "webcam"

    path = Path(source)

    if not path.exists():
        return "unknown"

    if path.is_dir():
        return "folder"

    ext = path.suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"

    return "unknown"


# ── Output path ───────────────────────────────────────────────────────

def make_output_path(results_dir: Path, source: str, suffix: str = "_detected") -> Path:
    """
    Build an output path in results_dir, preserving the original filename.
    Example: results/test_detected.jpg
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(source).stem
    ext = Path(source).suffix or ".jpg"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{stem}{suffix}_{timestamp}{ext}"
    return results_dir / filename


# ── Drawing ───────────────────────────────────────────────────────────

# 20 distinct BGR colours for class labels
COLORS = [
    (220, 82, 45), (45, 220, 82), (82, 45, 220), (220, 180, 45),
    (45, 82, 220), (180, 220, 45), (220, 45, 180), (45, 220, 180),
    (180, 45, 220), (220, 120, 45), (45, 120, 220), (120, 220, 45),
    (220, 45, 120), (120, 45, 220), (45, 220, 120), (200, 200, 45),
    (45, 200, 200), (200, 45, 200), (100, 200, 45), (45, 100, 200),
]


def get_color(class_id: int):
    return COLORS[class_id % len(COLORS)]


def draw_detections(frame: np.ndarray, result, names: dict) -> np.ndarray:
    """
    Draw bounding boxes and labels on a frame.
    Accepts a YOLOv8 result object.
    """
    img = frame.copy()

    if result.boxes is None:
        return img

    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        color = get_color(cls_id)
        label = f"{names.get(cls_id, cls_id)} {conf:.2f}"

        # Bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)

        # Label text
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return img


# ── Console output ────────────────────────────────────────────────────

def print_summary(result):
    """Print a clean detection summary to stdout."""
    names = result.names
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        print("[INFO] No objects detected.")
        return

    counts = {}
    for box in boxes:
        cls_id = int(box.cls[0])
        label = names.get(cls_id, str(cls_id))
        counts[label] = counts.get(label, 0) + 1

    lines = ", ".join(f"{v}× {k}" for k, v in sorted(counts.items()))
    print(f"[INFO] Detected: {lines}  (total: {len(boxes)})")


# ── COCO class names (80 classes) ────────────────────────────────────

COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard", 37: "surfboard",
    38: "tennis racket", 39: "bottle", 40: "wine glass", 41: "cup",
    42: "fork", 43: "knife", 44: "spoon", 45: "bowl", 46: "banana",
    47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli", 51: "carrot",
    52: "hot dog", 53: "pizza", 54: "donut", 55: "cake", 56: "chair",
    57: "couch", 58: "potted plant", 59: "bed", 60: "dining table",
    61: "toilet", 62: "tv", 63: "laptop", 64: "mouse", 65: "remote",
    66: "keyboard", 67: "cell phone", 68: "microwave", 69: "oven",
    70: "toaster", 71: "sink", 72: "refrigerator", 73: "book", 74: "clock",
    75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier",
    79: "toothbrush",
}
