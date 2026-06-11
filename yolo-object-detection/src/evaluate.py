"""
evaluate.py — Evaluate a YOLOv8 model on a validation set

Usage:
    python src/evaluate.py --model models/best.pt --data data/dataset.yaml
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Evaluation")

    parser.add_argument("--model", type=str, required=True,
                        help="Path to model weights (.pt)")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to dataset YAML")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--split", type=str, default="val",
                        choices=["val", "test"],
                        help="Dataset split to evaluate on")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--save-json", action="store_true",
                        help="Save results as JSON (COCO format)")
    parser.add_argument("--plots", action="store_true",
                        help="Save evaluation plots (confusion matrix, PR curve…)")

    return parser.parse_args()


def main():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    args = parse_args()

    model_path = ROOT / args.model
    data_path  = ROOT / args.data

    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        sys.exit(1)

    if not data_path.exists():
        print(f"[ERROR] Dataset config not found: {data_path}")
        sys.exit(1)

    print("=" * 50)
    print("  YOLOv8 Evaluation")
    print("=" * 50)
    print(f"  Model   : {model_path}")
    print(f"  Dataset : {data_path}")
    print(f"  Split   : {args.split}")
    print(f"  Device  : {args.device}")
    print("=" * 50)

    model = YOLO(str(model_path))

    metrics = model.val(
        data=str(data_path),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split=args.split,
        conf=args.conf,
        iou=args.iou,
        save_json=args.save_json,
        plots=args.plots,
    )

    # ── Print results ────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  Evaluation Results")
    print("=" * 50)
    print(f"  mAP@0.50       : {metrics.box.map50:.4f}")
    print(f"  mAP@0.50:0.95  : {metrics.box.map:.4f}")
    print(f"  Precision      : {metrics.box.mp:.4f}")
    print(f"  Recall         : {metrics.box.mr:.4f}")
    print("=" * 50)

    # Per-class breakdown
    if hasattr(metrics.box, "ap_class_index"):
        print("\n  Per-class AP@0.50:")
        names = model.names
        for i, cls_idx in enumerate(metrics.box.ap_class_index):
            ap = metrics.box.ap50[i]
            print(f"    {names[int(cls_idx)]:<20} {ap:.4f}")

    print()


if __name__ == "__main__":
    main()
