"""
detect.py — YOLO Object Detection
Supports: images, videos, folders, webcam (index 0)

Usage:
    python src/detect.py --source data/images/test.jpg
    python src/detect.py --source data/videos/test.mp4 --save
    python src/detect.py --source 0 --show
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import (
    draw_detections,
    get_source_type,
    make_output_path,
    print_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Object Detection")

    parser.add_argument("--source", type=str, required=True,
                        help="Image, video, folder path, or webcam index (0)")
    parser.add_argument("--model", type=str, default="models/yolov8n.pt",
                        help="Path to YOLO model weights (.pt)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="IoU threshold for NMS (default: 0.45)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Inference image size (default: 640)")
    parser.add_argument("--classes", nargs="+", type=int, default=None,
                        help="Filter by class IDs, e.g. --classes 0 2 3")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: 'cpu', '0' (GPU), 'mps'")
    parser.add_argument("--save", action="store_true",
                        help="Save annotated output to results/")
    parser.add_argument("--show", action="store_true",
                        help="Display results in a window")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file (overridden by CLI args)")

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML config and return detection section."""
    config_file = ROOT / config_path
    if config_file.exists():
        with open(config_file) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("detection", {})
    return {}


def run_detection(args):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    # ── Load model ────────────────────────────────────────────────────
    model_path = ROOT / args.model
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        print("  → Run: python scripts/download_model.py --model yolov8n")
        sys.exit(1)

    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(str(model_path))

    source = args.source
    source_type = get_source_type(source)
    print(f"[INFO] Source: {source}  (type: {source_type})")

    # ── Route to correct handler ───────────────────────────────────────
    if source_type == "webcam":
        run_webcam(model, args)
    elif source_type == "video":
        run_video(model, source, args)
    elif source_type == "image":
        run_image(model, source, args)
    elif source_type == "folder":
        run_folder(model, source, args)
    else:
        print(f"[ERROR] Unrecognised source: {source}")
        sys.exit(1)


# ── Image ─────────────────────────────────────────────────────────────

def run_image(model, source: str, args):
    results = model.predict(
        source=source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        classes=args.classes,
        device=args.device,
        verbose=False,
    )
    result = results[0]
    annotated = result.plot()

    print_summary(result)

    if args.show:
        cv2.imshow("YOLO Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if args.save:
        out_path = make_output_path(ROOT / "results", source)
        cv2.imwrite(str(out_path), annotated)
        print(f"[INFO] Saved → {out_path}")


# ── Folder (batch) ────────────────────────────────────────────────────

def run_folder(model, source: str, args):
    folder = Path(source)
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in folder.iterdir() if p.suffix.lower() in image_exts]

    if not images:
        print(f"[WARNING] No images found in {folder}")
        return

    print(f"[INFO] Found {len(images)} image(s) in {folder}")
    total_detections = 0

    for img_path in sorted(images):
        results = model.predict(
            source=str(img_path),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            classes=args.classes,
            device=args.device,
            verbose=False,
        )
        result = results[0]
        total_detections += len(result.boxes)

        if args.save:
            annotated = result.plot()
            out_path = make_output_path(ROOT / "results", str(img_path))
            cv2.imwrite(str(out_path), annotated)

        print(f"  {img_path.name}: {len(result.boxes)} detection(s)")

    print(f"[INFO] Done. Total detections: {total_detections}")
    if args.save:
        print(f"[INFO] Results saved in: {ROOT / 'results'}")


# ── Video ─────────────────────────────────────────────────────────────

def run_video(model, source: str, args):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {source}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if args.save:
        out_path = make_output_path(ROOT / "results", source, suffix="_out.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        print(f"[INFO] Writing to → {out_path}")

    frame_idx = 0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            source=frame,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            classes=args.classes,
            device=args.device,
            verbose=False,
        )
        annotated = results[0].plot()

        elapsed = time.time() - t_start
        current_fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0
        cv2.putText(annotated, f"FPS: {current_fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if writer:
            writer.write(annotated)

        if args.show:
            cv2.imshow("YOLO Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[INFO] Interrupted by user.")
                break

        frame_idx += 1
        if total_frames > 0:
            pct = 100 * frame_idx / total_frames
            print(f"\r[INFO] Processing frame {frame_idx}/{total_frames} ({pct:.1f}%)", end="")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"\n[INFO] Processed {frame_idx} frame(s) in {time.time()-t_start:.1f}s")


# ── Webcam ─────────────────────────────────────────────────────────────

def run_webcam(model, args):
    cam_index = int(args.source) if args.source.isdigit() else 0
    cap = cv2.VideoCapture(cam_index)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam (index {cam_index})")
        sys.exit(1)

    print("[INFO] Webcam started — press Q to quit")
    t_start = time.time()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            source=frame,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            classes=args.classes,
            device=args.device,
            verbose=False,
            stream=True,
        )
        for result in results:
            annotated = result.plot()

        elapsed = time.time() - t_start
        fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0
        cv2.putText(annotated, f"FPS: {fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YOLO — Webcam (Q to quit)", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[INFO] Session ended. {frame_idx} frames processed.")


if __name__ == "__main__":
    args = parse_args()

    # Merge config defaults (CLI args take priority)
    cfg = load_config(args.config)
    for key, val in cfg.items():
        if key in vars(args) and getattr(args, key) == parser_defaults(key):
            setattr(args, key, val)

    run_detection(args)


def parser_defaults(key):
    """Return argparse defaults for merging with config."""
    defaults = {
        "model": "models/yolov8n.pt",
        "conf": 0.25,
        "iou": 0.45,
        "imgsz": 640,
        "device": "cpu",
        "classes": None,
    }
    return defaults.get(key)
