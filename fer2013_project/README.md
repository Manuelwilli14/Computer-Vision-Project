# Facial Emotion Recognition — FER-2013

A complete pipeline for training a CNN on the FER-2013 dataset and running real-time emotion recognition via webcam.

## Project structure

```
fer2013_project/
├── data/                   # Place fer2013.csv here
├── models/                 # Saved .h5 models
├── outputs/                # Plots, confusion matrices
├── utils/
│   ├── data_loader.py      # Dataset loading & preprocessing
│   ├── augmentation.py     # ImageDataGenerator config
│   └── visualization.py    # Plotting helpers
├── notebooks/
│   └── exploration.ipynb   # EDA notebook
├── train.py                # Full training script
├── evaluate.py             # Evaluation & confusion matrix
├── webcam_demo.py          # Real-time webcam inference
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

Download [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) from Kaggle and place `fer2013.csv` inside the `data/` folder.

## Usage

### 1. Train
```bash
python train.py
```

### 2. Evaluate
```bash
python evaluate.py --model models/best_model.h5
```

### 3. Webcam demo
```bash
python webcam_demo.py --model models/best_model.h5
```

## Expected results

| Metric | Value |
|---|---|
| Val accuracy | ~63–68% |
| Inference (CPU) | < 30ms/frame |
| Input size | 48×48 grayscale |
| Classes | 7 emotions |

## Emotion classes

`0 Angry` · `1 Disgust` · `2 Fear` · `3 Happy` · `4 Neutral` · `5 Sad` · `6 Surprise`
