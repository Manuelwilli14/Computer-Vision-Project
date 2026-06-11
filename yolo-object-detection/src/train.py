"""
train.py — Train a YOLOv8 model on a custom dataset

Usage:
    python src/train.py --config config.yaml
    python src/train.py --data data/dataset.yaml --epochs 50 --batch 8
"""

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Training")

    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--model", type=str, default=None,
                        help="Base model (e.g. yolov8n.pt). Overrides config.")
    parser.add_argument("--data", type=str, default=None,
                        help="Dataset YAML path. Overrides config.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--name", type=str, default=None,
                        help="Run name (saved under runs/train/<name>)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume interrupted training")

    return parser.parse_args()


def load_train_config(config_path: str) -> dict:
    cfg_file = ROOT / config_path
    if cfg_file.exists():
        with open(cfg_file) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("training", {})
    print(f"[WARNING] Config not found: {cfg_file}. Using defaults.")
    return {}


def main():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    args = parse_args()
    cfg = load_train_config(args.config)

    # CLI args override config
    model_name  = args.model   or cfg.get("model", "yolov8n.pt")
    data_yaml   = args.data    or cfg.get("data", "data/dataset.yaml")
    epochs      = args.epochs  or cfg.get("epochs", 100)
    batch       = args.batch   or cfg.get("batch", 16)
    imgsz       = args.imgsz   or cfg.get("imgsz", 640)
    device      = args.device  or cfg.get("device", "cpu")
    patience    = cfg.get("patience", 50)
    save_period = cfg.get("save_period", 10)
    project     = cfg.get("project", "runs/train")
    name        = args.name    or cfg.get("name", "exp")

    data_path = ROOT / data_yaml
    if not data_path.exists():
        print(f"[ERROR] Dataset config not found: {data_path}")
        print("  → See README for dataset format requirements.")
        sys.exit(1)

    print("=" * 50)
    print("  YOLOv8 Training")
    print("=" * 50)
    print(f"  Base model  : {model_name}")
    print(f"  Dataset     : {data_path}")
    print(f"  Epochs      : {epochs}")
    print(f"  Batch size  : {batch}")
    print(f"  Image size  : {imgsz}")
    print(f"  Device      : {device}")
    print(f"  Output      : {project}/{name}")
    print("=" * 50)

    model = YOLO(model_name)

    results = model.train(
        data=str(data_path),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        patience=patience,
        save_period=save_period,
        project=project,
        name=name,
        resume=args.resume,
        verbose=True,
    )

    print("\n[INFO] Training complete!")
    best = Path(project) / name / "weights" / "best.pt"
    if best.exists():
        print(f"[INFO] Best weights saved → {best}")
    else:
        print(f"[INFO] Check {project}/{name}/weights/ for saved checkpoints.")


if __name__ == "__main__":
    main()
