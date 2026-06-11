"""
prepare_dataset.py — Validate and prepare a YOLO-format dataset

Usage:
    python scripts/prepare_dataset.py --dataset path/to/dataset --output data/
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a YOLO-format dataset")

    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the dataset folder (with images/ and labels/)")
    parser.add_argument("--output", type=str, default="data/",
                        help="Output directory (default: data/)")
    parser.add_argument("--split", type=float, default=0.2,
                        help="Validation split ratio (default: 0.2)")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Class names in order, e.g. --classes cat dog car")
    parser.add_argument("--check", action="store_true",
                        help="Only validate the dataset, do not copy files")

    return parser.parse_args()


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_pairs(images_dir: Path, labels_dir: Path):
    """Return (image_path, label_path) pairs that both exist."""
    pairs = []
    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        lbl = labels_dir / (img.stem + ".txt")
        if lbl.exists():
            pairs.append((img, lbl))
        else:
            print(f"  [WARNING] Missing label for: {img.name}")
    return pairs


def validate_labels(pairs):
    errors = 0
    for _, lbl in pairs:
        with open(lbl) as f:
            for i, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"  [ERROR] {lbl.name} line {i}: expected 5 fields, got {len(parts)}")
                    errors += 1
                    continue
                try:
                    cls_id = int(parts[0])
                    coords = [float(p) for p in parts[1:]]
                except ValueError:
                    print(f"  [ERROR] {lbl.name} line {i}: invalid values")
                    errors += 1
                    continue
                if any(v < 0 or v > 1 for v in coords):
                    print(f"  [ERROR] {lbl.name} line {i}: coordinates out of [0,1] range")
                    errors += 1
    return errors


def copy_split(pairs, out_images: Path, out_labels: Path):
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    for img, lbl in pairs:
        shutil.copy2(img, out_images / img.name)
        shutil.copy2(lbl, out_labels / lbl.name)


def write_dataset_yaml(output_dir: Path, classes: list):
    yaml_path = output_dir / "dataset.yaml"
    data = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(classes),
        "names": classes,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    print(f"[INFO] Dataset YAML written → {yaml_path}")
    return yaml_path


def main():
    args = parse_args()

    dataset_dir = Path(args.dataset)
    images_dir  = dataset_dir / "images"
    labels_dir  = dataset_dir / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        print(f"[ERROR] Expected {images_dir} and {labels_dir} to exist.")
        sys.exit(1)

    print(f"[INFO] Scanning dataset: {dataset_dir}")
    pairs = find_pairs(images_dir, labels_dir)
    print(f"[INFO] Found {len(pairs)} image–label pairs")

    if not pairs:
        print("[ERROR] No valid pairs found. Check your dataset structure.")
        sys.exit(1)

    print("[INFO] Validating labels…")
    errors = validate_labels(pairs)
    if errors:
        print(f"[ERROR] {errors} label error(s) found. Fix them before continuing.")
        if not args.check:
            sys.exit(1)
    else:
        print("[INFO] All labels valid ✓")

    if args.check:
        print("[INFO] Check complete (--check flag: no files copied).")
        return

    # ── Train / val split ─────────────────────────────────────────────
    random.shuffle(pairs)
    split_idx = int(len(pairs) * (1 - args.split))
    train_pairs = pairs[:split_idx]
    val_pairs   = pairs[split_idx:]

    print(f"[INFO] Split: {len(train_pairs)} train / {len(val_pairs)} val")

    output_dir = ROOT / args.output
    copy_split(train_pairs, output_dir / "images/train", output_dir / "labels/train")
    copy_split(val_pairs,   output_dir / "images/val",   output_dir / "labels/val")

    # ── Class names ───────────────────────────────────────────────────
    classes = args.classes or detect_classes(output_dir / "labels/train")
    if not classes:
        print("[WARNING] Could not detect class names. Use --classes to specify them.")
        classes = [f"class_{i}" for i in range(80)]

    yaml_path = write_dataset_yaml(output_dir, classes)

    print("\n[INFO] Dataset ready!")
    print(f"  Train images : {len(train_pairs)}")
    print(f"  Val images   : {len(val_pairs)}")
    print(f"  Classes ({len(classes)}): {', '.join(classes)}")
    print(f"\n  Use with: python src/train.py --data {yaml_path.relative_to(ROOT)}")


def detect_classes(labels_dir: Path) -> list:
    """Infer class IDs from label files and return a placeholder list."""
    class_ids = set()
    for lbl in labels_dir.glob("*.txt"):
        with open(lbl) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    try:
                        class_ids.add(int(parts[0]))
                    except ValueError:
                        pass
    if not class_ids:
        return []
    n = max(class_ids) + 1
    return [f"class_{i}" for i in range(n)]


if __name__ == "__main__":
    main()
