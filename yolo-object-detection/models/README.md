# models/

Pretrained and custom-trained YOLO weights go here.

## Download pretrained weights

```bash
python scripts/download_model.py --model yolov8n
```

## Available variants

| Model    | Params | Speed | mAP@0.50:0.95 |
|----------|--------|-------|----------------|
| yolov8n  | 3.2M   | ⚡⚡⚡⚡  | 37.3           |
| yolov8s  | 11M    | ⚡⚡⚡   | 44.9           |
| yolov8m  | 25M    | ⚡⚡    | 50.2           |
| yolov8l  | 43M    | ⚡     | 52.9           |
| yolov8x  | 68M    | 🐢     | 53.9           |

> `.pt` files are excluded from version control via `.gitignore`.
> Use Git LFS if you need to version large model files.
