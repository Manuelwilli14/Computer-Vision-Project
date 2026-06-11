"""
download_model.py — Download YOLOv8 pretrained weights

Usage:
    python scripts/download_model.py --model yolov8n
    python scripts/download_model.py --model yolov8s --output models/
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

AVAILABLE_MODELS = {
    "yolov8n": "Nano   (~3.2M params)  — fastest, least accurate",
    "yolov8s": "Small  (~11M params)   — good balance for most tasks",
    "yolov8m": "Medium (~25M params)   — better accuracy, moderate speed",
    "yolov8l": "Large  (~43M params)   — high accuracy",
    "yolov8x": "XLarge (~68M params)   — highest accuracy, slowest",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Download YOLOv8 pretrained weights")

    parser.add_argument("--model", type=str, default="yolov8n",
                        choices=list(AVAILABLE_MODELS.keys()),
                        help="Model variant to download (default: yolov8n)")
    parser.add_argument("--output", type=str, default="models/",
                        help="Directory to save the weights (default: models/)")
    parser.add_argument("--list", action="store_true",
                        help="List all available models and exit")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.list:
        print("\nAvailable YOLOv8 models:")
        for name, desc in AVAILABLE_MODELS.items():
            print(f"  {name:<12} {desc}")
        print()
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    output_dir = ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    model_filename = f"{args.model}.pt"
    dest = output_dir / model_filename

    if dest.exists():
        print(f"[INFO] Model already exists: {dest}")
        return

    print(f"[INFO] Downloading {args.model}...")
    print(f"       {AVAILABLE_MODELS[args.model]}")

    # ultralytics downloads automatically on first load
    model = YOLO(model_filename)

    # Move from CWD to models/
    downloaded = Path(model_filename)
    if downloaded.exists() and not dest.exists():
        downloaded.rename(dest)

    if dest.exists():
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"[INFO] Saved → {dest}  ({size_mb:.1f} MB)")
    else:
        print(f"[INFO] Model cached by ultralytics. Copy to {output_dir} if needed.")


if __name__ == "__main__":
    main()
